import os
from dotenv import load_dotenv
import wandb
import mpc_imle.utils as utils
from datetime import datetime

load_dotenv()

use_wandb = os.getenv("WANDB", "FALSE").upper() == "TRUE"

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str
    config: str

args = Parser().parse_args('imle')

# Initialize W&B only if enabled
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
        name=f"imle_{args.dataset}_H{args.horizon}_D{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        entity=os.getenv("WANDB_ENTITY"),
        mode="online"
    )

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
    reward_weighting=args.reward_weighting,
    beta=args.beta,
    exp_min=args.exp_min,
    exp_max=args.exp_max,
    seed=args.seed,
    max_samples=args.max_samples,
    sampling_type=args.sampling_type
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
    cond_dim=args.cond_factor * observation_dim,
    dim_mults=args.dim_mults,
    dim=args.z_dim,
    latent_dim=args.latent_dim,
    attention=args.attention,
    device=args.device,
    use_layer_norm=args.use_layer_norm
)

imle_config = utils.Config(
    args.imle,
    savepath=(args.savepath, 'imle_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    sample_factor=args.sample_factor,
    noise_coef=args.noise_coef,
    loss_type=args.loss_type,
    alpha=args.alpha,
    offset=args.offset,
    latent_dim=args.latent_dim,
    device=args.device,
    eps_radius=args.eps_radius,
    rs=args.rs,
    eps_threshold=args.eps_threshold
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
    val_freq=args.val_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    staleness=args.staleness,
    latent_dim=args.latent_dim,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    horizon=args.horizon,
    use_imle=True,
    ada_imle=args.ada_imle,
    warmup_steps=args.warmup_steps,
    use_adamw=args.use_adamw,
    total_training_steps=args.n_train_steps
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

if args.device:
    model = model.to(args.device)

imle = imle_config(model)
if args.device:
    imle = imle.to(args.device)

trainer = trainer_config(imle, dataset, renderer, wandb_run=wandb_run)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
idx, batch = dataset[0]

batch = utils.batchify(batch)
latent = imle.generate_latent(*batch)
loss, _ = imle.loss(*batch, latent)
loss.backward()
print('✓')

# #-----------------------------------------------------------------------------#
# #--------------------------------- main loop ---------------------------------#
# #-----------------------------------------------------------------------------#

trainer.train(max_train_steps=args.n_train_steps, max_epochs=200)

trainer.save(args.n_train_steps)
if wandb_run is not None:
    wandb_run.finish()
