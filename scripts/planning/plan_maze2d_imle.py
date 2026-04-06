import json
import numpy as np
from os.path import join

from mpc_imle.guides.policies import Policy
import mpc_imle.datasets as datasets
import mpc_imle.utils as utils
from dotenv import load_dotenv

load_dotenv()

class Parser(utils.Parser):
    dataset: str = "maze2d-umaze-v1"
    config: str = "config.maze2d_imle_jax"


# ---------------------------------- setup ---------------------------------- #

args = Parser().parse_args("plan")
env = datasets.load_environment(args.dataset)

# ---------------------------------- loading ---------------------------------- #

diffusion_experiment = utils.load_experiment(
    args.logbase,
    args.dataset,
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
    generator="imle_config.pkl",
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

# ---------------------------------- main loop ---------------------------------- #

observation = env.reset()

if args.conditional:
    print("Resetting target")
    env.set_target()

# set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0], dtype=np.float32),
}

# observations for rendering
rollout = [observation.copy()]

total_reward = 0.0
score = 0.0
terminal = False

for t in range(env.max_episode_steps):
    state = env.state_vector().copy()

    if t == 0:
        cond[0] = observation

        action, samples = policy(cond, batch_size=args.batch_size)

        # JAX safety: renderer + indexing code expects mutable / numpy arrays
        # Policy might return JAX arrays; convert once here.
        actions = np.array(samples.actions)            # [B, H, act_dim]
        observations = np.array(samples.observations)  # [B, H, obs_dim]

        actions0 = actions[0]
        sequence = observations[0]

    if t < len(sequence) - 1:
        next_waypoint = sequence[t + 1].copy()
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0.0

    # can use actions or define a simple controller based on state predictions
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

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f"{t}.png")

        if t == 0:
            # JAX safety: pass numpy, not JAX arrays, because renderer mutates in-place
            renderer.composite(fullpath, observations, ncol=1)

        renderer.composite(
            join(args.savepath, "rollout.png"),
            np.array(rollout, dtype=np.float32)[None],
            ncol=1,
        )

    if terminal:
        break

    observation = next_observation

# save result as a json file
json_path = join(args.savepath, "rollout.json")
json_data = {
    "score": float(score),
    "step": int(t),
    "return": float(total_reward),
    "term": bool(terminal),
    "epoch_diffusion": int(diffusion_experiment.epoch),
}
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2, sort_keys=True)
