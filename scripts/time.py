import argparse
import subprocess
import sys

SCRIPT_MAP = {
    # locomotion
    ("jax", "diffusion", "locomotion"): "scripts/timing/time_guided_policy_jax.py",
    ("torch", "diffusion", "locomotion"): "scripts/timing/time_guided_policy.py",
    ("jax", "imle", "locomotion"): "scripts/timing/time_guided_policy_jax.py",
    ("torch", "imle", "locomotion"): "scripts/timing/time_guided_policy.py",

    # maze
    ("jax", "diffusion", "maze2d"): "scripts/timing/time_maze_jax.py",
    ("torch", "diffusion", "maze2d"): "scripts/timing/time_maze2d.py",
    ("jax", "imle", "maze2d"): "scripts/timing/time_maze_jax.py",
    ("torch", "imle", "maze2d"): "scripts/timing/time_maze2d.py",
}

CONFIG_MAP = {
    # locomotion
    ("jax", "imle", "locomotion"): "config.locomotion_imle_jax",
    ("jax", "diffusion", "locomotion"): "config.locomotion_diffusion_jax",
    ("torch", "imle", "locomotion"): "config.locomotion_imle",
    ("torch", "diffusion", "locomotion"): "config.locomotion_diffusion",

    # maze2d
    ("jax", "imle", "maze2d"): "config.maze2d_imle_jax",
    ("jax", "diffusion", "maze2d"): "config.maze2d_diffusion_jax",
    ("torch", "imle", "maze2d"): "config.maze2d_imle",
    ("torch", "diffusion", "maze2d"): "config.maze2d_diffusion",
}

DATASETS = [
    # Maze2D
    "maze2d-large-v1", "maze2d-medium-v1", "maze2d-umaze-v1",

    # Locomotion
    "walker2d-medium-v2", "walker2d-medium-replay-v2", "walker2d-medium-expert-v2",
    "hopper-medium-v2", "hopper-medium-replay-v2", "hopper-medium-expert-v2",
    "halfcheetah-medium-v2", "halfcheetah-medium-replay-v2", "halfcheetah-medium-expert-v2",
]

def get_domain(dataset: str) -> str:
    if dataset.startswith("maze2d"):
        return "maze2d"
    if dataset.startswith(("walker2d", "hopper", "halfcheetah")):
        return "locomotion"
    raise SystemExit(f"Unknown dataset domain for: {dataset}")

def main():
    p = argparse.ArgumentParser(add_help=True)

    p.add_argument("--backend", choices=["jax", "torch"], default="torch")
    p.add_argument("--model", choices=["imle", "diffusion"], required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)

    args, extra = p.parse_known_args()

    domain = get_domain(args.dataset)
    key = (args.backend, args.model, domain)

    if key not in SCRIPT_MAP:
        raise SystemExit(
            f"Unsupported timing combo:\n"
            f"  backend={args.backend}, model={args.model}, domain={domain}\n"
            f"Supported: {sorted(SCRIPT_MAP.keys())}"
        )

    script = SCRIPT_MAP[key]

    cmd = [sys.executable, script, "--config", CONFIG_MAP[key], "--dataset", args.dataset] + extra

    print("Running timing:\n  " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
