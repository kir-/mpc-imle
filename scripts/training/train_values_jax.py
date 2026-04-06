import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from dotenv import load_dotenv
import wandb

import mpc_imle.utils as utils

load_dotenv()
use_wandb = os.getenv("WANDB", "FALSE").upper() == "TRUE"

class Parser(utils.Parser):
    dataset: str
    config: str

args = Parser().parse_args("values")

wandb_run = None
if "WANDB_SWEEP_ID" in os.environ:
    print("[train_values_jax.py] Detected stray WANDB_SWEEP_ID — clearing to run as single experiment.")
    os.environ.pop("WANDB_SWEEP_ID", None)
    os.environ.pop("WANDB_CONFIG", None)
    os.environ.pop("WANDB_RUN_ID", None)

if use_wandb:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_run = wandb.init(
        project="mpc-imle",
        name=f"values_jax_{args.dataset}_H{args.horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        entity=os.getenv("WANDB_ENTITY"),
        mode="online",
    )

# ---------------- dataset ----------------

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    # value-specific
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, "render_config.pkl"),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

# ---------------- model + value diffusion ----------------

model_config = utils.Config(
    args.model, 
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    dim=getattr(args, "z_dim", 32),
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=getattr(args, "clip_denoised", True),  # if used
    predict_epsilon=getattr(args, "predict_epsilon", True),  # ignored in value diffusion
)

trainer_config = utils.Config(
    utils.TrainerJax,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    val_freq=args.val_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    n_reference=getattr(args, "n_reference", 8),
    n_samples=getattr(args, "n_samples", 2),
    horizon=args.horizon,
    use_imle=False,
    latent_dim=0,
    staleness=1,
    is_value_fn=True,
)

value_model = model_config()
value_diffusion = diffusion_config(generator=value_model)

trainer = trainer_config(value_diffusion, dataset, renderer, wandb_run=wandb_run)

# ---------------- smoke test ----------------

utils.report_parameters_jax(trainer.params)

print("Testing forward...", end=" ", flush=True)
idx, batch = dataset[0]
batch = utils.batchify_jax(batch)

x = jnp.array(np.array(batch[0]))
cond = batch[1]
target = jnp.array(np.array(batch[2]))

rng = random.PRNGKey(0)
rng, loss_key = random.split(rng)

loss, infos = value_diffusion.apply(
    {"params": trainer.params},
    loss_key,
    x,
    cond,
    None,
    None,
    target=target,
    method=value_diffusion.loss,
)

# grad test
def loss_wrap(p, key):
    l, _ = value_diffusion.apply({"params": p}, key, x, cond, None, None, target=target, method=value_diffusion.loss)
    return l

rng, grad_key = random.split(rng)
_ = jax.grad(lambda p: loss_wrap(p, grad_key))(trainer.params)

print("✓")

# ---------------- train ----------------

trainer.train(max_train_steps=args.n_train_steps, max_epochs=100)
trainer.save(args.n_train_steps)
if wandb_run is not None:
    wandb_run.finish()
