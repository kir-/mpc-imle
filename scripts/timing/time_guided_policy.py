import os, time, statistics as stats
from os.path import join
import numpy as np
import torch
from mpc_imle.utils.timer import TIMER_LOGGER, Timer_Better
import atexit
torch.set_default_dtype(torch.float32)
try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass
from dotenv import load_dotenv

import mpc_imle.sampling as sampling
import mpc_imle.utils as utils

# ------------------------ device sync helpers ------------------------ #
def sync_device_from_module(module):
    try:
        dev = next(module.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

def debug_model_dtypes(model, name):
    """Debug function to check model parameter dtypes"""
    print(f"\n=== {name} Parameter Types ===")
    for param_name, param in model.named_parameters():
        print(f"{param_name}: dtype={param.dtype}, device={param.device}")
        if 'blocks.0' in param_name:
            break

# ----------------------------- main --------------------------------- #
def main():
    load_dotenv()

    class Parser(utils.Parser):
        dataset: str = "walker2d-medium-replay-v2"
        config: str  = "config.locomotion_imle"
        sigma: float = 1.0
        timing_warmup: int = 3
        timing_runs: int = 20
        savepath: str = "timings"
        tag: str = ""

        # optionally print shapes from the last call
        print_samples: bool = False

    args = Parser().parse_args("plan")
    args.sample_impl = "imle" if "imle" in args.config else "nstep"

    def _slug(s: str) -> str:
        return str(s).replace(os.sep, "_").replace("/", "_").replace(" ", "_")

    safe_dataset = _slug(args.dataset)
    safe_impl    = _slug(args.sample_impl)
    safe_tag     = _slug(args.tag or os.path.basename(args.diffusion_loadpath) or "run")
    csv_path     = join(args.savepath, f"timings__{safe_dataset}__{safe_impl}__b{args.batch_size}__{safe_tag}.csv")

    run_meta_fixed = {
        "dataset": args.dataset,
        "sample_impl": args.sample_impl,
        "batch_size": int(args.batch_size),
    }

    atexit.register(lambda: TIMER_LOGGER.flush_to_csv(csv_path, extra=run_meta_fixed))

    # -------------------- deterministic-ish backend -------------------- #
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")  # PyTorch 2.x
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
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

    os.makedirs(args.savepath, exist_ok=True)

    # --------------------------- loading --------------------------- #
    gen_cfg_file = "imle_config.pkl" if args.sample_impl.lower() == "imle" else "diffusion_config.pkl"
    diffusion_experiment = utils.load_experiment(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, seed=args.seed,
        generator=gen_cfg_file,
    )
    value_experiment = utils.load_experiment(
        args.loadbase, args.dataset, args.value_loadpath,
        epoch=args.value_epoch, seed=args.seed,
    )
    utils.check_compatibility(diffusion_experiment, value_experiment)

    # -------- device & dtype --------
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    DTYPE = torch.float32
    print(f"[INFO] Using device: {device_str}")

    diffusion = diffusion_experiment.ema
    diffusion = diffusion.to(device)   # device first
    diffusion = diffusion.float()      # then dtype
    diffusion = diffusion.eval()

    if hasattr(diffusion, "sigma"):
        diffusion.sigma = getattr(args, "sigma", 1.0)

    dataset = diffusion_experiment.dataset

    value_function = value_experiment.ema
    value_function = value_function.to(device)
    value_function = value_function.float()
    value_function = value_function.eval()

    # Debug: Check parameter types after loading
    debug_model_dtypes(diffusion, "Diffusion Model")
    debug_model_dtypes(value_function, "Value Function")

    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()

    # choose guided sampling rule
    if args.sample_impl.lower() == "imle":
        sample_fn = sampling.imle_sample
    elif args.sample_impl.lower() == "nstep":
        sample_fn = sampling.n_step_guided_p_sample
    else:
        raise ValueError(f"Unknown sample_impl: {args.sample_impl} (use 'imle' or 'nstep')")

    # build the guided policy wrapper
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        sample_fn=sample_fn,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        verbose=False,
    )
    policy = policy_config()

    # Force the policy's diffusion/guide models to match device/dtype
    if hasattr(policy, 'diffusion_model'):
        policy.diffusion_model = policy.diffusion_model.to(device).float().eval()
    if hasattr(policy, 'guide') and hasattr(policy.guide, 'model'):
        policy.guide.model = policy.guide.model.to(device).float().eval()

    # ------------------------- build condition ------------------------- #
    env = dataset.env
    reset_out = env.reset()
    observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    observation = observation.astype(np.float32)

    # For locomotion: condition only on t=0 observation.
    conditions = {0: observation}
    dev = next(diffusion.parameters()).device
    conditions_t = utils.to_torch(conditions, dtype=torch.float32, device=dev)
    conditions_t = {k: v.float() for k, v in conditions_t.items()}  # keep fp32

    # Debug: Check condition tensor types
    print("\n=== Condition Tensor Types ===")
    for k, v in conditions_t.items():
        print(f"conditions_t[{k}]: dtype={v.dtype}, device={v.device}, shape={v.shape}")

    # --------------------------- warmup --------------------------- #
    print("\n=== Starting Warmup ===")
    for i in range(max(0, int(args.timing_warmup))):
        print(f"Warmup run {i+1}/{args.timing_warmup}")
        sync_device_from_module(diffusion)
        try:
            _ = policy(conditions_t, batch_size=args.batch_size, verbose=False)
            print(f"Warmup run {i+1} successful")
        except Exception as e:
            print(f"Warmup run {i+1} failed: {e}")
            raise e
        sync_device_from_module(diffusion)

    # ---------------------------- timing ---------------------------- #
    print(f"\n=== Starting {args.timing_runs} Timing Runs ===")
    timings = []
    per_run = []

    for i in range(max(1, int(args.timing_runs))):
        TIMER_LOGGER.clear() 
        t_plan = Timer_Better(device, name=f"plan_total/b{args.batch_size}")
        _, samples = policy(conditions_t, batch_size=args.batch_size, verbose=False)
        sec = t_plan() 
        timings.append(sec)

        # --- compute per-run totals BEFORE flushing ---
        gen_s   = TIMER_LOGGER.total("generator")
        guid_s  = TIMER_LOGGER.total("guidance")
        rank_s  = TIMER_LOGGER.total("ranking") or TIMER_LOGGER.total("rank")
        plan_s  = TIMER_LOGGER.total(f"plan_total/b{args.batch_size}")

        per_run.append({
            "run": i + 1,
            "generator_ms": gen_s * 1e3,
            "guidance_ms":  guid_s * 1e3,
            "ranking_ms":   rank_s * 1e3,
            "plan_ms":      plan_s * 1e3,
        })

        print(
            f"Run {i+1}/{args.timing_runs}: "
            f"gen={per_run[-1]['generator_ms']:.2f} ms  "
            f"guid={per_run[-1]['guidance_ms']:.2f} ms  "
            f"rank={per_run[-1]['ranking_ms']:.2f} ms  "
            f"plan={per_run[-1]['plan_ms']:.2f} ms",
            flush=True,
        )

        hzn = int(getattr(diffusion, "horizon", -1))
        TIMER_LOGGER.flush_to_csv(
            csv_path,
            extra={**run_meta_fixed, "horizon": hzn, "timestamp": int(time.time()), "run_idx": i}
        )
    # --------------------------- summary --------------------------- #
    def metric_summary(values):
        if not values:
            return "N/A"
        m = stats.mean(values)
        s = stats.pstdev(values) if len(values) > 1 else 0.0
        md = stats.median(values)
        return f"mean={m:.2f} ms  std={s:.2f} ms  median={md:.2f} ms"

    # Build metric arrays from per_run
    gen_vals   = [r["generator_ms"] for r in per_run]
    guid_vals  = [r["guidance_ms"]  for r in per_run]
    rank_vals  = [r["ranking_ms"]   for r in per_run]
    plan_vals  = [r["plan_ms"]      for r in per_run]

    print("\n=== Per-metric averages over all runs ===")
    print(f"generator: {metric_summary(gen_vals)}")
    print(f"guidance:  {metric_summary(guid_vals)}")
    print(f"ranking:   {metric_summary(rank_vals)}")
    print(f"plan:      {metric_summary(plan_vals)}")

if __name__ == "__main__":
    main()