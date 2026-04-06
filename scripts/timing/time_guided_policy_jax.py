import os, time, statistics as stats, atexit
from os.path import join

import numpy as np
import jax
import jax.numpy as jnp
from dotenv import load_dotenv

import mpc_imle.sampling as sampling
import mpc_imle.utils as utils

from mpc_imle.utils.timer import TIMER_LOGGER

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

load_dotenv()


class Parser(utils.Parser):
    dataset: str = "walker2d-medium-v2"
    config: str = "config.locomotion_imle_jax"
    timing_warmup: int = 3
    timing_runs: int = 20
    savepath: str = "timings_jax"
    tag: str = ""


args = Parser().parse_args("plan")
args.jax_guide = True

args.sample_impl = "imle" if "imle" in args.config else "nstep"
generator = "imle_config.pkl" if "imle" in args.config else "diffusion_config.pkl"
def _slug(s: str) -> str:
    return str(s).replace(os.sep, "_").replace("/", "_").replace(" ", "_")


safe_dataset = _slug(args.dataset)
safe_impl = _slug(args.sample_impl)
safe_tag = _slug(args.tag or os.path.basename(args.diffusion_loadpath) or "run")

csv_path = join(
    args.savepath,
    f"timings_jax__{safe_dataset}__{safe_impl}__b{args.batch_size}__{safe_tag}.csv",
)
os.makedirs(args.savepath, exist_ok=True)

run_meta_fixed = {
    "dataset": args.dataset,
    "sample_impl": args.sample_impl,
    "batch_size": int(args.batch_size),
}

atexit.register(lambda: TIMER_LOGGER.flush_to_csv(csv_path, extra=run_meta_fixed))

jax.config.update("jax_disable_jit", False)
jax.config.update("jax_default_prng_impl", "threefry2x32")
os.environ["JAX_ENABLE_X64"] = "0"

print(f"[INFO] Using JAX devices: {jax.devices()}")

# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

diffusion_experiment = utils.load_experiment(
    args.loadbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    seed=args.seed,
    generator=generator,
    jax=args.jax,
)
value_experiment = utils.load_experiment(
    args.loadbase,
    args.dataset,
    args.value_loadpath,
    epoch=args.value_epoch,
    seed=args.seed,
    jax=args.jax_guide,
)

utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset

value_function = value_experiment.ema
if args.jax_guide:
    guide_config = utils.Config(
        args.guide, 
        apply_fn=value_function.apply,
        params=value_experiment.trainer.ema_params,
        verbose=False
    )
else:
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

if args.sample_impl.lower() == "imle":
    sample_fn = sampling.imle_sample
elif args.sample_impl.lower() == "nstep":
    sample_fn = sampling.n_step_guided_p_sample
else:
    raise ValueError(f"Unknown sample_impl: {args.sample_impl} (use 'imle' or 'nstep')")

base_kwargs = dict(
    guide=guide,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    sample_fn=sample_fn,
    model=value_function, 
    rank=True,
    do_guide=True,
    use_jax_guide=args.jax_guide,
    scale=getattr(args, "scale", None),
    n_guide_steps=getattr(args, "n_guide_steps", None),
    t_stopgrad=getattr(args, "t_stopgrad", None),
    scale_grad_by_std=getattr(args, "scale_grad_by_std", None),
    verbose=False,
)

if args.jax:
    base_kwargs["ema_params"] = diffusion_experiment.trainer.ema_params
    base_kwargs["action_dim"] = diffusion.action_dim

policy_config = utils.Config(args.policy, **base_kwargs)
policy = policy_config()

# -----------------------------------------------------------------------------#
# -------------------------------- build condition ----------------------------#
# -----------------------------------------------------------------------------#

env = dataset.env
reset_out = env.reset()
observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
observation = observation.astype(np.float32)

conditions = {0: observation}
conditions_jax = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in conditions.items()}

print("\n=== Condition Tensor Types ===")
for k, v in conditions_jax.items():
    print(f"conditions_jax[{k}]: dtype={v.dtype}, shape={v.shape}")

# -----------------------------------------------------------------------------#
# ------------------------------ warmup & timing ------------------------------#
# -----------------------------------------------------------------------------#

print("\n=== Starting Warmup ===")
for i in range(max(0, int(args.timing_warmup))):
    print(f"Warmup run {i+1}/{args.timing_warmup}", flush=True)
    try:
        # policy handles TIMER_LOGGER.start_run/end_run internally
        _ = policy(conditions_jax, batch_size=args.batch_size, verbose=False)
        print(f"Warmup run {i+1} successful")
    except Exception as e:
        print(f"Warmup run {i+1} failed: {e}")
        raise

print(f"\n=== Starting {args.timing_runs} Timing Runs ===")
per_run = []

for i in range(max(1, int(args.timing_runs))):
    # policy handles timing + blocking internally (gen/guid/rank/plan)
    _ = policy(conditions_jax, batch_size=args.batch_size, verbose=False)
    gen_s = TIMER_LOGGER.total("generator")
    guid_s = TIMER_LOGGER.total("guidance")
    rank_s = TIMER_LOGGER.total("ranking") or TIMER_LOGGER.total("rank")
    plan_s = TIMER_LOGGER.total("plan")

    per_run.append(
        {
            "run": i + 1,
            "generator_ms": gen_s * 1e3,
            "guidance_ms": guid_s * 1e3,
            "ranking_ms": rank_s * 1e3,
            "plan_ms": plan_s * 1e3,
        }
    )

    print(
        f"Run {i+1}/{args.timing_runs}: "
        f"gen={per_run[-1]['generator_ms']:.2f} ms  "
        f"guid={per_run[-1]['guidance_ms']:.2f} ms  "
        f"rank={per_run[-1]['ranking_ms']:.2f} ms  "
        f"plan={per_run[-1]['plan_ms']:.2f} ms",
        flush=True,
    )

    # Flush this run’s *samples* to CSV immediately (same as Torch)
    hzn = int(getattr(diffusion, "horizon", -1))
    TIMER_LOGGER.flush_to_csv(
        csv_path,
        extra={
            **run_meta_fixed,
            "horizon": hzn,
            "timestamp": int(time.time()),
            "run_idx": i,
        },
    )

# -----------------------------------------------------------------------------#
# --------------------------------- summary -----------------------------------#
# -----------------------------------------------------------------------------#


def metric_summary(values):
    if not values:
        return "N/A"
    m = stats.mean(values)
    s = stats.pstdev(values) if len(values) > 1 else 0.0
    md = stats.median(values)
    return f"mean={m:.2f} ms  std={s:.2f} ms  median={md:.2f} ms"


gen_vals = [r["generator_ms"] for r in per_run]
guid_vals = [r["guidance_ms"] for r in per_run]
rank_vals = [r["ranking_ms"] for r in per_run]
plan_vals = [r["plan_ms"] for r in per_run]

print("\n=== Per-metric averages over all runs ===")
print(f"generator: {metric_summary(gen_vals)}")
print(f"guidance:  {metric_summary(guid_vals)}")
print(f"ranking:   {metric_summary(rank_vals)}")
print(f"plan:      {metric_summary(plan_vals)}")
