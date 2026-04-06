import argparse
import subprocess
import sys

SCRIPT_MAP = {
    ("policy", "jax", "diffusion"): "scripts/training/train_diffusion_jax.py", 
    ("policy", "torch", "diffusion"): "scripts/training/train_diffusion.py", 
    ("policy", "jax", "imle"):  "scripts/training/train_imle_jax.py",
    ("policy", "jax", "imle_inv"):  "scripts/training/train_imle_inv.py",
    ("policy", "torch", "imle"): "scripts/training/train_imle.py",
    ("value", "jax", "diffusion"): "scripts/training/train_values_jax.py",
    ("value", "torch", "diffusion"): "scripts/training/train_values.py",
}

CONFIG_MAP = {
    # policy configs - locomotion
    ("policy", "jax", "imle", "locomotion"): "config.locomotion_imle_jax",
    ("policy", "jax", "diffusion", "locomotion"): "config.locomotion_diffusion_jax",
    ("policy", "torch", "imle", "locomotion"): "config.locomotion_imle",
    ("policy", "torch", "diffusion", "locomotion"): "config.locomotion_diffusion",

    # policy configs - maze2d
    ("policy", "jax", "imle", "maze2d"): "config.maze2d_imle_jax",
    ("policy", "jax", "diffusion", "maze2d"): "config.maze2d_diffusion_jax",
    ("policy", "torch", "imle", "maze2d"): "config.maze2d_imle",
    ("policy", "torch", "diffusion", "maze2d"): "config.maze2d_diffusion",

    # value configs - locomotion
    ("value", "jax", "diffusion", "locomotion"): "config.locomotion_diffusion_jax",
    ("value", "torch", "diffusion", "locomotion"): "config.locomotion_diffusion",
}

DATASETS = [
    # Maze2D
    "maze2d-large-v1", "maze2d-medium-v1", "maze2d-umaze-v1",
    
    # Locomotion
    "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2",
    "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
    "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
]

def get_domain(dataset):
    """Determine domain type from dataset name."""
    if dataset.startswith("maze2d"):
        return "maze2d"
    elif dataset.startswith(("walker2d", "hopper", "halfcheetah")):
        return "locomotion"

def main():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--task", choices=["policy", "value"], default="policy")
    p.add_argument("--backend", choices=["jax", "torch"], default="torch")
    p.add_argument("--model", choices=["imle", "imle_inv", "diffusion"], required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)

    args, extra = p.parse_known_args()

    domain = get_domain(args.dataset)
    key_cfg = (args.task, args.backend, args.model, domain)

    if key_cfg not in CONFIG_MAP:
        raise SystemExit(
            f"Unsupported combo: task={args.task}, backend={args.backend}, model={args.model}, domain={domain}\n"
            f"Supported: {sorted(CONFIG_MAP.keys())}"
        )

    key_script = (args.task, args.backend, args.model)
    if key_script not in SCRIPT_MAP:
        raise SystemExit(
            f"Missing script mapping for task={args.task}, backend={args.backend}, model={args.model}\n"
            f"Known: {sorted(SCRIPT_MAP.keys())}"
        )

    script = SCRIPT_MAP[key_script]
    config = CONFIG_MAP[key_cfg]

    cmd = [sys.executable, script, "--config", config, "--dataset", args.dataset] + extra
    
    print("Running:\n  " + " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()