import os, time, statistics as stats, atexit
from os.path import join

import numpy as np
import jax
import jax.numpy as jnp
from dotenv import load_dotenv

import mpc_imle.sampling as sampling
import mpc_imle.utils as utils
from mpc_imle.utils.timer import TIMER_LOGGER

load_dotenv()

class Parser(utils.Parser):
    dataset: str = "maze2d-medium-v1"
    config: str = "config.maze2d_imle_jax"

    # timing knobs
    timing_warmup: int = 3
    timing_runs: int = 20
    savepath: str = "timings_jax"
    tag: str = ""

args = Parser().parse_args("plan")
args.jax = True

def _slug(s: str) -> str:
    return str(s).replace(os.sep, "_").replace("/", "_").replace(" ", "_")

safe_dataset = _slug(args.dataset)
safe_tag = _slug(args.tag or os.path.basename(args.diffusion_loadpath) or "run")

csv_path = join(
    args.savepath,
    f"timings_jax__{safe_dataset}__unguided__b{args.batch_size}__{safe_tag}.csv",
)
os.makedirs(args.savepath, exist_ok=True)

run_meta_fixed = {
    "dataset": args.dataset,
    "mode": "unguided",
    "batch_size": int(args.batch_size),
}

atexit.register(lambda: TIMER_LOGGER.flush_to_csv(csv_path, extra=run_meta_fixed))

jax.config.update("jax_disable_jit", False)
jax.config.update("jax_default_prng_impl", "threefry2x32")
os.environ["JAX_ENABLE_X64"] = "0"

print(f"[INFO] Using JAX devices: {jax.devices()}")

# ---------------- loading (ONLY diffusion) ---------------- #

diffusion_experiment = utils.load_experiment(
    args.loadbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    seed=args.seed,
    generator = (
        "imle_config.pkl"
        if "imle" in args.config
        else "diffusion_config.pkl"
    ),
    jax=True,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset

base_kwargs = dict(
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    verbose=False
)

if args.jax:
    base_kwargs["ema_params"] = diffusion_experiment.trainer.ema_params
    base_kwargs["action_dim"] = diffusion.action_dim

policy_config = utils.Config(args.policy, **base_kwargs)
policy = policy_config()

# ---------------- build condition ---------------- #

env = dataset.env
reset_out = env.reset()
observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
observation = observation.astype(np.float32)

H = int(getattr(diffusion, "horizon", getattr(args, "horizon", 32)))

# Maze2D usually uses start+goal conditioning (two 4D vectors -> 8D total)
start = observation.astype(np.float32)
goal = start.copy()
goal_xy = start[2:4].copy()
goal[0:2] = goal_xy
goal[2:4] = goal_xy

conditions = {0: start, H - 1: goal}
conditions_jax = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in conditions.items()}

print("\n=== Condition ===")
print("keys:", sorted(conditions_jax.keys()))
print("shapes:", {k: tuple(v.shape) for k, v in conditions_jax.items()})

def _block_until_ready(action, samples):
    try:
        if hasattr(action, "block_until_ready"):
            action = action.block_until_ready()
    except Exception:
        pass
    for attr in ["actions", "observations", "values"]:
        try:
            v = getattr(samples, attr, None)
            if hasattr(v, "block_until_ready"):
                _ = v.block_until_ready()
        except Exception:
            pass

def metric_summary(values):
    if not values:
        return "N/A"
    m = stats.mean(values)
    s = stats.pstdev(values) if len(values) > 1 else 0.0
    md = stats.median(values)
    return f"mean={m:.2f} ms  std={s:.2f} ms  median={md:.2f} ms"

# ---------------- warmup & timing ---------------- #

print("\n=== Starting Warmup ===")
for i in range(max(0, int(args.timing_warmup))):
    print(f"Warmup run {i+1}/{args.timing_warmup}", flush=True)

    # optional: keep run boundaries for consistency
    TIMER_LOGGER.start_run()

    t0 = time.perf_counter()
    action, samples = policy(conditions_jax, batch_size=args.batch_size, verbose=False)
    _block_until_ready(action, samples)
    dt_ms = (time.perf_counter() - t0) * 1e3

    TIMER_LOGGER.end_run()
    print(f"Warmup {i+1}: plan={dt_ms:.2f} ms", flush=True)

print(f"\n=== Starting {args.timing_runs} Timing Runs ===")
per_run = []

for i in range(max(1, int(args.timing_runs))):
    TIMER_LOGGER.start_run()

    t0 = time.perf_counter()
    action, samples = policy(conditions_jax, batch_size=args.batch_size, verbose=False)
    _block_until_ready(action, samples)
    plan_ms = (time.perf_counter() - t0) * 1e3

    TIMER_LOGGER.end_run()

    # For unguided Maze2D, "generator" == "plan" in terms of what you care about.
    row = {
        "run": i + 1,
        "generator_ms": plan_ms,
        "guidance_ms": 0.0,
        "ranking_ms": 0.0,
        "plan_ms": plan_ms,
    }
    per_run.append(row)

    print(
        f"Run {i+1}/{args.timing_runs}: "
        f"plan={row['plan_ms']:.2f} ms",
        flush=True,
    )

    # Flush to CSV: store your manual measured ms in extra fields
    hzn = int(getattr(diffusion, "horizon", -1))
    TIMER_LOGGER.flush_to_csv(
        csv_path,
        extra={
            **run_meta_fixed,
            "horizon": hzn,
            "timestamp": int(time.time()),
            "run_idx": i,
            "manual_plan_ms": plan_ms,
            "manual_generator_ms": plan_ms,
        },
    )

print("\n=== Averages over all runs ===")
print(f"plan:      {metric_summary([r['plan_ms'] for r in per_run])}")
print(f"generator: {metric_summary([r['generator_ms'] for r in per_run])}")
