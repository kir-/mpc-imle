import os
import copy
import pickle
import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import optax

from torch.utils.data import Subset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from .arrays import concatenate_cond_jax
from .timer import Timer

use_wandb = os.getenv("WANDB", "FALSE").upper() == "TRUE"
DEVICE = os.getenv("DEVICE_SPECIFIC")


class JaxDataLoader:
    """Pure JAX/NumPy dataloader replacement for PyTorch DataLoader."""

    def __init__(self, dataset, batch_size=32, shuffle=True, rng_key=None, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
        self.length = len(dataset)

        if self.drop_last:
            self.num_batches = self.length // self.batch_size
        else:
            self.num_batches = (self.length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.length)

        if self.shuffle:
            self.rng_key, subkey = random.split(self.rng_key)
            indices = jax.random.permutation(subkey, indices)

        for batch_idx in range(self.num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.length)

            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]
            batch_data = [self.dataset[int(i)] for i in batch_indices]
            ids, batches = zip(*batch_data)

            trajectories = jnp.asarray(np.stack([b.trajectories for b in batches]))
            values = jnp.asarray(np.stack([b.values for b in batches]))

            conditions = {}
            if getattr(batches[0], "conditions", None):
                for k in batches[0].conditions.keys():
                    conditions[k] = jnp.asarray(np.stack([b.conditions[k] for b in batches]))

            yield np.asarray(ids), trajectories, conditions, values

    def __len__(self):
        return self.num_batches


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def _ema_update(ema_params, params, beta: float):
    beta = jnp.asarray(beta, jnp.float32)
    return jax.tree_util.tree_map(lambda e, p: beta * e + (1.0 - beta) * p, ema_params, params)


class TrainerJax:
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        val_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder="./results",
        n_reference=8,
        n_samples=2,
        val_size=100,
        wandb_run=None,
        use_imle=False,
        is_value_fn=False,
        horizon=10,
        staleness=1,
        latent_dim=6,
        bucket=None,
        warmup_steps=0,
        use_adamw=False,
        weight_decay=1e-4,  # only used if use_adamw=True
    ):
        super().__init__()

        self.model = diffusion_model
        self.renderer = renderer

        self.ema_decay = float(ema_decay)
        self.step_start_ema = int(step_start_ema)
        self.update_ema_every = int(update_ema_every)

        self.log_freq = int(log_freq)
        self.sample_freq = int(sample_freq)
        self.save_freq = int(save_freq)
        self.val_freq = int(val_freq)
        self.label_freq = int(label_freq)
        self.save_parallel = bool(save_parallel)

        self.batch_size = int(train_batch_size)
        self.gradient_accumulate_every = int(gradient_accumulate_every)

        self.is_value_fn = bool(is_value_fn)
        
        self.use_imle = bool(use_imle)
        self.latent_dim = int(latent_dim)
        self.staleness = int(staleness)
        self.horizon = int(horizon)

        self.use_adamw = bool(use_adamw)
        self.weight_decay = float(weight_decay)

        # Split dataset into train/val
        dataset_size = len(dataset)
        val_size = min(int(val_size), dataset_size)
        train_size = dataset_size - val_size
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)

        # dataloaders
        self.rng = random.PRNGKey(0)
        self.dataloader = cycle(
            JaxDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, rng_key=self.rng)
        )
        self.val_dataloader = cycle(
            JaxDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, rng_key=self.rng)
        )
        self.dataloader_vis = cycle(
            JaxDataLoader(self.train_dataset, batch_size=1, shuffle=True, rng_key=self.rng)
        )

        self.logdir = results_folder
        self.n_reference = int(n_reference)
        self.n_samples = int(n_samples)

        self.step = 0
        self.wandb_run = wandb_run

        # init params + opt state + jit step
        self._init_training_state(train_lr=float(train_lr), warmup_steps=int(warmup_steps))

    # ----------------------------
    # init optimizer + params
    # ----------------------------
    def _init_training_state(self, train_lr: float, warmup_steps: int):
        # Peek a batch to infer shapes (init uses batch_size=1)
        _, x, cond, rewards = next(self.dataloader)
        x = jnp.asarray(x)
        rewards = jnp.asarray(rewards)

        B0 = 1
        H = int(x.shape[1])
        D = int(x.shape[2])

        dummy_x = jnp.zeros((B0, H, D), dtype=jnp.float32)
        dummy_cond = {k: jnp.zeros((B0, v.shape[-1]), dtype=jnp.float32) for k, v in cond.items()}
        dummy_rewards = jnp.zeros((B0,), dtype=jnp.float32)
        dummy_target = jnp.zeros((B0,), dtype=jnp.float32)
        dummy_key = random.PRNGKey(123)

        self.rng, init_rng = random.split(self.rng)

        latent_dim = getattr(self.model, "latent_dim", 0)
        dummy_latent = jnp.zeros((B0, H, latent_dim), dtype=jnp.float32)

        variables = self.model.init(
            init_rng,
            dummy_key,      # key
            dummy_x,        # x
            dummy_cond,     # cond
            dummy_rewards,  # rewards
            dummy_latent,   # latent (IMLE uses, diffusion ignores)
            dummy_target, 
            method=self.model.loss,
        )

        self.params = variables["params"]

        self.base_lr = float(train_lr)
        self.warmup_steps = int(warmup_steps)

        # Optax lr schedule: linear warmup -> constant
        if self.warmup_steps > 0:
            self.lr_schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(0.0, self.base_lr, self.warmup_steps),
                    optax.constant_schedule(self.base_lr),
                ],
                boundaries=[self.warmup_steps],
            )
        else:
            self.lr_schedule = optax.constant_schedule(self.base_lr)

        # Optimizer
        if self.use_adamw:
            self.tx = optax.adamw(learning_rate=self.lr_schedule, weight_decay=self.weight_decay)
        else:
            self.tx = optax.adam(learning_rate=self.lr_schedule)

        self.opt_state = self.tx.init(self.params)

        # EMA params
        self.ema_params = copy.deepcopy(self.params)

        # build JIT'd step (ONE top-level jit)
        self._train_step = jax.jit(
            self._train_step_fn,
            static_argnames=("use_imle",),
        )

    # ----------------------------
    # one jitted train step
    # ----------------------------
    def _train_step_fn(self, params, opt_state, ema_params, rng, step_i, x, cond, rewards, use_imle: bool):
        """
        One JIT'd step: latent gen (optional) + loss + grads + optax update + EMA.

        NOTE:
        - params is the raw params pytree
        - EMA gate uses step_i (int32 scalar) to avoid Python state inside jit
        """

        def make_latent(rng_in):
            rng_in, subkey = random.split(rng_in)
            z = self.model.apply(
                {"params": params},
                subkey,
                x,
                cond,
                rewards,
                method=self.model.generate_latent,
            )
            return rng_in, z

        def make_random_latent(rng_in):
            rng_in, subkey = random.split(rng_in)
            z = random.normal(subkey, (x.shape[0], self.model.horizon, self.model.latent_dim))
            return rng_in, z

        rng, latent = jax.lax.cond(
            use_imle,
            lambda r: make_latent(r),
            lambda r: make_random_latent(r),
            rng,
        )
        rng, loss_key = random.split(rng)

        def loss_fn(p):
            if self.is_value_fn:
                # here `rewards` actually holds the target from dataloader
                target = rewards
                loss, infos = self.model.apply(
                    {"params": p},
                    loss_key,
                    x,
                    cond,
                    None,      # rewards unused
                    None,      # latent unused
                    target=target,
                    method=self.model.loss,
                )
            else:
                loss, infos = self.model.apply(
                    {"params": p},
                    loss_key,
                    x,
                    cond,
                    rewards,
                    latent,
                    method=self.model.loss,
                )
            return loss, infos

        (loss, infos), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        updates, new_opt_state = self.tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # EMA: before step_start_ema, track params directly; after that do exponential average
        def ema_before(_):
            return new_params

        def ema_after(_):
            return _ema_update(ema_params, new_params, beta=self.ema_decay)

        new_ema_params = jax.lax.cond(step_i < self.step_start_ema, ema_before, ema_after, operand=None)

        return new_params, new_opt_state, new_ema_params, loss, infos, rng

    # ----------------------------
    # train loop
    # ----------------------------
    def train(self, max_train_steps, max_epochs):
        timer = Timer()

        train_size = len(self.train_dataset)
        num_batches_per_epoch = max(1, train_size // self.batch_size)

        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1} / {max_epochs} | {self.logdir}")

            for _ in tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch+1}", unit="batch"):
                # Build “mega-batch” by concatenation
                xs, rs = [], []
                cond_accum = {}
                for _acc in range(self.gradient_accumulate_every):
                    _, x, cond, rewards = next(self.dataloader)
                    xs.append(jnp.asarray(x))
                    for k, v in cond.items():
                        cond_accum.setdefault(k, []).append(jnp.asarray(v))
                    rs.append(jnp.asarray(rewards))

                x_big = jnp.concatenate(xs, axis=0)
                cond_big = {k: jnp.concatenate(vs, axis=0) for k, vs in cond_accum.items()}
                rewards_big = jnp.concatenate(rs, axis=0)

                self.rng, subkey = random.split(self.rng)

                step_i = jnp.asarray(self.step, dtype=jnp.int32)

                (self.params, self.opt_state, self.ema_params,
                 step_loss, step_infos, self.rng) = self._train_step(
                    self.params,
                    self.opt_state,
                    self.ema_params,
                    subkey,
                    step_i,
                    x_big,
                    cond_big,
                    rewards_big,
                    self.use_imle,
                )

                if self.step % self.save_freq == 0:
                    self.save(self.step)

                if self.step % self.log_freq == 0:
                    infos_str = " | ".join([f"{k}: {float(v):8.4f}" for k, v in step_infos.items()])
                    print(f"{self.step}: {float(step_loss):8.4f} | {infos_str} | t: {timer():8.4f}")

                if self.wandb_run:
                    if self.step % self.val_freq == 0:
                        val_loss = self.compute_validation_loss()
                        self.wandb_run.log(
                            {"validation_loss": val_loss, "train_loss": float(step_loss), "step": self.step}
                        )
                    elif self.step % self.log_freq == 0:
                        self.wandb_run.log({"train_loss": float(step_loss), "step": self.step})

                self.step += 1
                if self.step >= max_train_steps:
                    print("Maximum Training Steps Hit:", max_train_steps)
                    return

    # ----------------------------
    # validation
    # ----------------------------
    def compute_validation_loss(self):
        total_val_loss = 0.0
        num_batches = 0

        n_batches = min(10, max(1, len(self.val_dataset) // max(1, self.batch_size)))
        for _ in range(n_batches):
            _, x, cond, rewards = next(self.val_dataloader)

            x = jnp.asarray(x)
            rewards = jnp.asarray(rewards)

            self.rng, subkey = random.split(self.rng)

            if self.use_imle:
                latent = self.model.apply(
                    {"params": self.ema_params},
                    subkey,
                    x,
                    cond,
                    rewards,
                    method=self.model.generate_latent,
                )
            else:
                latent = random.normal(subkey, (x.shape[0], self.model.horizon, self.model.latent_dim))

            self.rng, key = random.split(self.rng)
            if self.is_value_fn:
                target = rewards 
                loss, _ = self.model.apply(
                    {"params": self.ema_params},
                    key,
                    x,
                    cond,
                    None,
                    None,
                    target=target,
                    method=self.model.loss,
                )
            else:
                loss, _ = self.model.apply(
                    {"params": self.ema_params},
                    key,
                    x,
                    cond,
                    rewards,
                    latent,
                    method=self.model.loss,
                )

            total_val_loss += float(loss)
            num_batches += 1

        return total_val_loss / max(1, num_batches)

    # ----------------------------
    # checkpointing
    # ----------------------------
    def save(self, step):
        os.makedirs(self.logdir, exist_ok=True)
        data = {
            "step": self.step,
            "params": self.params,
            "ema_params": self.ema_params,
            "opt_state": self.opt_state,
            "use_adamw": self.use_adamw,
            "weight_decay": self.weight_decay,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
        }
        savepath = os.path.join(self.logdir, f"state_{int(step)}.pkl")
        with open(savepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[ utils/training ] Saved model to {savepath}", flush=True)

    def load(self, step):
        loadpath = os.path.join(self.logdir, f"state_{int(step)}.pkl")
        with open(loadpath, "rb") as f:
            data = pickle.load(f)

        self.step = data["step"]
        self.params = data["params"]
        self.ema_params = data["ema_params"]
        self.opt_state = data.get("opt_state", None)
        if self.opt_state is None:
            print("[load] opt_state missing — loading params only (inference).")

        # (optional) restore these
        self.use_adamw = bool(data.get("use_adamw", self.use_adamw))
        self.weight_decay = float(data.get("weight_decay", self.weight_decay))
        self.base_lr = float(data.get("base_lr", self.base_lr))
        self.warmup_steps = int(data.get("warmup_steps", self.warmup_steps))
