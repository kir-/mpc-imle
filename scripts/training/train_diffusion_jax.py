import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from dotenv import load_dotenv
import wandb

import mpc_imle.utils as utils

# Load environment variables from .env
load_dotenv()

use_wandb = os.getenv("WANDB", "FALSE").upper() == "TRUE"

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str
    config: str

args = Parser().parse_args("diffusion")

# Initialize W&B only if enabled
wandb_run = None

if "WANDB_SWEEP_ID" in os.environ:
    print("[train_diffusion_jax.py] Detected stray WANDB_SWEEP_ID — clearing to run as single experiment.")
    os.environ.pop("WANDB_SWEEP_ID", None)
    os.environ.pop("WANDB_CONFIG", None)
    os.environ.pop("WANDB_RUN_ID", None)

if use_wandb:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_run = wandb.init(
        project="mpc-imle",
        name=f"diffusion_jax_{args.dataset}_H{args.horizon}_D{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        entity=os.getenv("WANDB_ENTITY"),
        mode="online",
    )

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    reward_weighting=getattr(args, "reward_weighting", False),  # safe if missing
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

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=args.cond_factor * observation_dim if hasattr(args, "cond_factor") else observation_dim,
    dim_mults=args.dim_mults,
    dim=getattr(args, "z_dim", 32),  # safe default
    attention=args.attention,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
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
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    horizon=args.horizon,
    staleness=1,
    latent_dim=0,  
    use_imle=False,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

generator = model_config()
diffusion = diffusion_config(generator=generator)

trainer = trainer_config(diffusion, dataset, renderer, wandb_run=wandb_run)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters_jax(trainer.params)

print("Testing forward...", end=" ", flush=True)
idx, batch = dataset[0]

batch = utils.batchify_jax(batch)

x = jnp.array(np.array(batch[0]))
cond = batch[1]
rewards = jnp.array(np.array(batch[2]))

rng = random.PRNGKey(0)
rng, subkey = random.split(rng)

# call diffusion loss through apply
loss, infos, = diffusion.apply({"params": trainer.params}, subkey, x, cond, rewards, None, method=diffusion.loss)

# quick grad test
def loss_wrap(p, key):
    l, _, = diffusion.apply({"params": p}, key, x, cond, rewards, None, method=diffusion.loss)
    return l

rng, subkey = random.split(rng)
_ = jax.grad(lambda p: loss_wrap(p, subkey))(trainer.params)

print("✓")

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

trainer.train(max_train_steps=args.n_train_steps, max_epochs=100)
trainer.save(args.n_train_steps)

if wandb_run is not None:
    wandb_run.finish()
