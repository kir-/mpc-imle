import os, time, statistics as stats, atexit
from os.path import join
import numpy as np
import torch
from dotenv import load_dotenv
import mpc_imle.utils as utils
from mpc_imle.utils.timer import TIMER_LOGGER
from mpc_imle.guides.policies import Policy
load_dotenv()

# ------------------------ global torch knobs ------------------------ #
torch.set_default_dtype(torch.float32)
try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass

def _sync_device(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()

def sync_device_from_module(module):
    try:
        dev = next(module.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    _sync_device(dev)

def configure_torch_backend():
    # "deterministic-ish" backend (safe guards)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    try:
        # PyTorch 2.x
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

def metric_summary(values):
    if not values:
        return "N/A"
    m = stats.mean(values)
    s = stats.pstdev(values) if len(values) > 1 else 0.0
    md = stats.median(values)
    return f"mean={m:.2f} ms  std={s:.2f} ms  median={md:.2f} ms"

# ------------------------ args ------------------------ #
class Parser(utils.Parser):
    dataset: str = "maze2d-medium-v1"
    config: str = "config.maze2d_imle"

    # timing knobs
    timing_warmup: int = 3
    timing_runs: int = 20
    savepath: str = "timings_torch"
    tag: str = ""

    # optional
    profile: bool = False
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_active: int = 3
    profile_repeat: int = 1

args = Parser().parse_args("plan")
args.jax = False

def _slug(s: str) -> str:
    return str(s).replace(os.sep, "_").replace("/", "_").replace(" ", "_")

safe_dataset = _slug(args.dataset)
safe_tag = _slug(args.tag or os.path.basename(getattr(args, "diffusion_loadpath", "")) or "run")

csv_path = join(
    args.savepath,
    f"timings_torch__{safe_dataset}__unguided__b{args.batch_size}__{safe_tag}.csv",
)
os.makedirs(args.savepath, exist_ok=True)

run_meta_fixed = {
    "dataset": args.dataset,
    "mode": "unguided",
    "batch_size": int(args.batch_size),
}

atexit.register(lambda: TIMER_LOGGER.flush_to_csv(csv_path, extra=run_meta_fixed))

# ------------------------ device ------------------------ #
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[INFO] Using torch device: {device}")

configure_torch_backend()

# ---------------- loading (diffusion OR imle; your generator picker) ---------------- #
experiment = utils.load_experiment(
    args.loadbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    seed=args.seed,
    generator=("imle_config.pkl" if "imle" in args.config else "diffusion_config.pkl"),
    jax=False,
)

model = experiment.ema
dataset = experiment.dataset

# move model
try:
    model = model.to(device)
except Exception:
    pass
try:
    model.eval()
except Exception:
    pass

policy = Policy(model, dataset.normalizer)
try:
    policy = policy.to(device)
except Exception:
    pass


# ---------------- build condition ---------------- #
env = dataset.env
reset_out = env.reset()
observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
observation = observation.astype(np.float32)

H = int(getattr(model, "horizon", getattr(args, "horizon", 32)))

start = observation.astype(np.float32)
goal = start.copy()
goal_xy = start[2:4].copy()
goal[0:2] = goal_xy
goal[2:4] = goal_xy

conditions = {0: start, H - 1: goal}
conditions_torch = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in conditions.items()}

print("\n=== Condition ===")
print("keys:", sorted(conditions_torch.keys()))
print("shapes:", {k: tuple(v.shape) for k, v in conditions_torch.items()})

# ---------------- warmup & timing ---------------- #
print("\n=== Starting Warmup ===")
for i in range(max(0, int(args.timing_warmup))):
    print(f"Warmup run {i+1}/{args.timing_warmup}", flush=True)
    TIMER_LOGGER.start_run()

    _sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        action, samples = policy(conditions_torch, batch_size=args.batch_size)
    _sync_device(device)
    dt_ms = (time.perf_counter() - t0) * 1e3

    TIMER_LOGGER.end_run()
    print(f"Warmup {i+1}: plan={dt_ms:.2f} ms", flush=True)

print(f"\n=== Starting {args.timing_runs} Timing Runs ===")
per_run = []

def one_call():
    with torch.no_grad():
        return policy(conditions_torch, batch_size=args.batch_size)

# Optional profiler (kept lightweight, no need to use unless debugging)
if getattr(args, "profile", False):
    from torch.profiler import profile, ProfilerActivity, schedule

    prof_out = join(args.savepath, f"prof__{safe_dataset}__b{int(args.batch_size)}__{safe_tag}")
    os.makedirs(prof_out, exist_ok=True)
    print(f"[INFO] Profiling enabled. Traces in: {prof_out}")

    prof_sched = schedule(
        wait=int(args.profile_wait),
        warmup=int(args.profile_warmup),
        active=int(args.profile_active),
        repeat=int(args.profile_repeat),
    )

    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else []),
        schedule=prof_sched,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_out),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i in range(max(1, int(args.timing_runs))):
            TIMER_LOGGER.start_run()

            _sync_device(device)
            t0 = time.perf_counter()
            action, samples = one_call()
            _sync_device(device)
            plan_ms = (time.perf_counter() - t0) * 1e3

            TIMER_LOGGER.end_run()
            prof.step()

            row = dict(run=i + 1, generator_ms=plan_ms, guidance_ms=0.0, ranking_ms=0.0, plan_ms=plan_ms)
            per_run.append(row)

            print(f"Run {i+1}/{args.timing_runs}: plan={plan_ms:.2f} ms", flush=True)

            hzn = int(getattr(model, "horizon", -1))
            TIMER_LOGGER.flush_to_csv(
                csv_path,
                extra={
                    **run_meta_fixed,
                    "horizon": hzn,
                    "timestamp": int(time.time()),
                    "run_idx": i,
                    "manual_plan_ms": plan_ms,
                    "manual_generator_ms": plan_ms,
                    "device": str(device),
                },
            )
else:
    for i in range(max(1, int(args.timing_runs))):
        TIMER_LOGGER.start_run()

        _sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            action, samples = policy(conditions_torch, batch_size=args.batch_size)
        _sync_device(device)
        plan_ms = (time.perf_counter() - t0) * 1e3

        TIMER_LOGGER.end_run()

        row = dict(run=i + 1, generator_ms=plan_ms, guidance_ms=0.0, ranking_ms=0.0, plan_ms=plan_ms)
        per_run.append(row)

        print(f"Run {i+1}/{args.timing_runs}: plan={plan_ms:.2f} ms", flush=True)

        hzn = int(getattr(model, "horizon", -1))
        TIMER_LOGGER.flush_to_csv(
            csv_path,
            extra={
                **run_meta_fixed,
                "horizon": hzn,
                "timestamp": int(time.time()),
                "run_idx": i,
                "manual_plan_ms": plan_ms,
                "manual_generator_ms": plan_ms,
                "device": str(device),
            },
        )

print("\n=== Averages over all runs ===")
print(f"plan:      {metric_summary([r['plan_ms'] for r in per_run])}")
print(f"generator: {metric_summary([r['generator_ms'] for r in per_run])}")
