import json
import numpy as np
from os.path import join

import mpc_imle.datasets as datasets
import mpc_imle.utils as utils
from dotenv import load_dotenv

load_dotenv() 

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d_imle_jax'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_experiment(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed, generator='imle_config.pkl', jax=True
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

base_kwargs = dict(
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    verbose=False
)
base_kwargs["ema_params"] = diffusion_experiment.trainer.ema_params
base_kwargs["action_dim"] = diffusion.action_dim

policy_config = utils.Config(args.policy, **base_kwargs)
policy = policy_config()
#---------------------------------- main loop ----------------------------------#

observation = env.reset()

if getattr(args, "conditional", False):
    print("Resetting target")
    env.set_target()

# Set conditioning xy position to be the goal at final time
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0.0, 0.0], dtype=np.float32),
}

rollout = [observation.copy()]

total_reward = 0.0
score = 0.0
terminal = False

# These will be created at t==0 after planning
sequence = None          # NumPy: [H, obs_dim]
plan_obs_np = None       # NumPy: [B, H, obs_dim] for rendering

for t in range(env.max_episode_steps):
    state = env.state_vector().copy()

    # Plan once (open-loop). Replan if you want, but Maze2D often works open-loop.
    if t == 0:
        cond[0] = observation

        action0, samples = policy(cond, batch_size=args.batch_size, verbose=getattr(args, "verbose", True))

        # -------------------- CRITICAL JAX->NUMPY BOUNDARY -------------------- #
        # Anything downstream (waypoint mutation, renderer) must be NumPy.
        plan_actions_np = np.asarray(samples.actions)         # [B, H, act_dim]
        plan_obs_np = np.asarray(samples.observations)        # [B, H, obs_dim]
        plan_values_np = np.asarray(samples.values) if samples.values is not None else None

        # Use first batch element as the open-loop plan
        sequence = plan_obs_np[0]                             # [H, obs_dim] NumPy

    # Pick next waypoint from predicted sequence
    if t < len(sequence) - 1:
        next_waypoint = sequence[t + 1].copy()
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0.0  # <- safe because next_waypoint is NumPy

    # Simple controller based on predicted next state waypoint
    action = (next_waypoint[:2] - state[:2]) + (next_waypoint[2:] - state[2:])

    next_observation, reward, terminal, _ = env.step(action)
    total_reward += float(reward)
    score = env.get_normalized_score(total_reward)

    print(
        f"t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | "
        f"{action}"
    )

    if "maze2d" in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(f"maze | pos: {xy} | goal: {goal}")

    rollout.append(next_observation.copy())

    # Visualization / rendering
    if (t % args.vis_freq == 0) or terminal:
        fullpath = join(args.savepath, f"{t}.png")

        if t == 0:
            # renderer mutates arrays in-place -> must pass NumPy
            renderer.composite(fullpath, plan_obs_np, ncol=1)

        renderer.composite(
            join(args.savepath, "rollout.png"),
            np.asarray(rollout, dtype=np.float32)[None],
            ncol=1,
        )

    if terminal:
        break

    observation = next_observation

# ---------------------------------- save ---------------------------------- #
json_path = join(args.savepath, "rollout.json")
json_data = {
    "score": float(score),
    "step": int(t),
    "return": float(total_reward),
    "term": bool(terminal),
    "epoch_diffusion": int(getattr(diffusion_experiment, "epoch", -1)),
}
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2, sort_keys=True)

