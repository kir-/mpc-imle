import mpc_imle.utils as utils
import pdb
import os
import wandb
from dotenv import load_dotenv
from datetime import datetime

load_dotenv() 

use_wandb = os.getenv("WANDB", "FALSE").upper() == "TRUE"

wandb_run = None
if 'WANDB_SWEEP_ID' in os.environ:
    print("[train_imle.py] Detected stray WANDB_SWEEP_ID — clearing to run as single experiment.")
    os.environ.pop('WANDB_SWEEP_ID', None)
    os.environ.pop('WANDB_CONFIG', None)
    os.environ.pop('WANDB_RUN_ID', None)
    
if use_wandb:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_run = wandb.init(
        project="mpc-imle",
        name=f"diffusion_value_function_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        entity=os.getenv("WANDB_ENTITY"),
        mode="online"
    )


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('values')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    ## value-specific kwargs
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
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
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    n_reference=args.n_reference,
    value_function=True,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
if args.device:
    model = model.to(args.device)

diffusion = diffusion_config(model)
if args.device:
    diffusion = diffusion.to(args.device)

trainer = trainer_config(diffusion, dataset, renderer, wandb_run=wandb_run)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
idx, batch = dataset[0]

batch = utils.batchify(batch)
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

trainer.train(max_train_steps=args.n_train_steps, max_epochs=200)

trainer.save(args.n_train_steps)
if wandb_run is not None:
    wandb_run.finish()