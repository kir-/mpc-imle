from collections import namedtuple
import os
import time
import numpy as np
import torch

import jax
import jax.numpy as jnp
from jax import random
import einops

import mpc_imle.utils as utils
from mpc_imle.datasets.preprocessing import get_policy_preprocess_fn
from mpc_imle.models.helpers.sampling_jax import cond_to_tensor

from dotenv import load_dotenv

from mpc_imle.utils.timer import Timer_Better_Jax, TIMER_LOGGER
from mpc_imle.models.helpers.sampling_jax import sort_by_values 

load_dotenv()

Trajectories = namedtuple("Trajectories", "actions observations values")
DEVICE = os.getenv("DEVICE_SPECIFIC")


class GuidedPolicyJax:
    """
    Orchestrates:
      - JAX IMLE sampling
      - optional Torch guidance
      - optional JAX ranking
      - precise timing via Timer_Better_Jax
    """

    def __init__(
        self,
        guide,
        diffusion_model,
        normalizer,
        preprocess_fns,
        ema_params=None,
        action_dim=None,
        **sample_kwargs,
    ):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.ema_params = ema_params
        self.normalizer = normalizer
        self.action_dim = action_dim or diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)

        # extra knobs (guidance / ranking)
        self.sample_kwargs = dict(sample_kwargs)
        self.sample_fn = self.sample_kwargs.get("sample_fn", None)
        self.use_jax_guide = self.sample_kwargs.get("use_jax_guide", None)
        self.model = self.sample_kwargs.get("model", None)
        self.rank = bool(self.sample_kwargs.get("rank", True))
        self.do_guide_default = bool(self.sample_kwargs.get("do_guide", True))
        # RNG
        self.rng = random.PRNGKey(0)

        # -------------------- JIT apply for fast sampling -------------------- #

        def _unwrap_params(ema_params):
            vars_ = ema_params
            if isinstance(vars_, dict) and "params" in vars_:
                vars_ = vars_["params"]
                if isinstance(vars_, dict) and "params" in vars_:
                    vars_ = vars_["params"]
            return vars_

        self._unwrap_params = _unwrap_params

        self._jit_apply = jax.jit(
            lambda params, key, conds: self.diffusion_model.apply({"params": params}, key, conds)
        )

        if self.guide is not None:
            self._jit_guide = jax.jit(
                lambda x, cond: self.guide(
                    x,
                    cond,
                    jnp.zeros((x.shape[0],), dtype=jnp.int32),
                )
            )
        else:
            self._jit_guide = None

    # --------------------------------------------------------------------- #
    # main call
    # --------------------------------------------------------------------- #

    def __call__(self, conditions, batch_size=1, verbose=True, do_guide=None):
        if do_guide is None:
            do_guide = self.do_guide_default

        # fresh per-call timing buffer
        TIMER_LOGGER.start_run()

        # full-plan timer (wall + device)
        t_plan = Timer_Better_Jax(name="plan", autolog=True, echo=False)

        # -------------------- preprocess / format -------------------- #
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        # -------------------- JAX sampling -------------------- #
        self.rng, subkey = random.split(self.rng)

        variables = self._unwrap_params(self.ema_params)

        jax.tree_util.tree_leaves(conditions)[0].block_until_ready()

        t_gen = Timer_Better_Jax(name="generator", autolog=True, echo=False)
        samples = self._jit_apply(variables, subkey, conditions)
        t_gen(out=samples.trajectories)

        trajectories = samples.trajectories
        values = getattr(
            samples,
            "values",
            jnp.zeros((batch_size,), dtype=jnp.float32),
        )

        # -------------------- guidance + ranking -------------------- #
        if do_guide and self.sample_fn is not None and self.guide is not None:
            t_guid = Timer_Better_Jax(name="guidance", autolog=True, echo=False)
            values = self._jit_guide(trajectories, conditions)
            values.block_until_ready()
            t_guid(out=values)

            if self.rank:
                t_rank = Timer_Better_Jax(name="ranking", autolog=True, echo=False)
                trajectories, values = sort_by_values(trajectories, values)
                t_rank(out=trajectories)

        # -------------------- postprocess -------------------- #
        actions = trajectories[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(actions, "actions")
        action = jnp.asarray(actions[0, 0])

        normed_obs = trajectories[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_obs, "observations")

        # block on final outputs to close plan timing
        t_plan(out=actions)

        # -------------------- logging -------------------- #
        if verbose:
            gen_ms  = TIMER_LOGGER.total("generator") * 1e3
            guid_ms = TIMER_LOGGER.total("guidance") * 1e3
            rank_ms = TIMER_LOGGER.total("ranking") * 1e3
            plan_ms = TIMER_LOGGER.total("plan") * 1e3

            print(
                f"gen={gen_ms:.2f} ms  "
                f"guid={guid_ms:.2f} ms  "
                f"rank={rank_ms:.2f} ms  "
                f"plan={plan_ms:.2f} ms"
            )

        return action, Trajectories(actions, observations, values)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )
        conditions = jax.tree_util.tree_map(jnp.asarray, conditions)
        conditions = utils.apply_dict(
            lambda x: einops.repeat(x, "d -> repeat d", repeat=batch_size),
            conditions,
        )
        return conditions

class ModelPolicyJax:
    """
    Minimal JAX policy wrapper around a Flax model.

    Expected model.apply signature:
        samples = model.apply(variables, key, conds)
    where samples has:
        - samples.trajectories : [B, H, action_dim + obs_dim]
        - samples.values       : optional

    This wrapper:
      - preprocesses + normalizes + repeats conditions
      - calls model.apply (jitted)
      - unnormalizes outputs
      - returns (first_action, Trajectories)
    """

    def __init__(
        self,
        diffusion_model,
        normalizer,
        ema_params=None,
        action_dim=None,
        seed=0,
        jit=True,
    ):
        self.model = diffusion_model
        self.normalizer = normalizer
        self.ema_params = ema_params
        self.action_dim = action_dim or diffusion_model.action_dim

        self.rng = random.PRNGKey(seed)

        def _unwrap_params(ema_params):
            vars_ = ema_params
            if isinstance(vars_, dict) and "params" in vars_:
                vars_ = vars_["params"]
                if isinstance(vars_, dict) and "params" in vars_:
                    vars_ = vars_["params"]
            return vars_

        self._unwrap_params = _unwrap_params

        # IMPORTANT: Flax .apply is NOT jitted by default; jit it once.
        self._jit_apply = jax.jit(
            lambda params, key, conds: self.model.apply({"params": params}, key, conds)
        )

    def __call__(self, conditions, batch_size=1, verbose=True):
        # preprocess
        conditions = {k: v for k, v in conditions.items()}
        # normalize + repeat
        conditions = self._format_conditions(conditions, batch_size)

        # rng + params
        self.rng, subkey = random.split(self.rng)

        variables = self.ema_params
        if variables is None:
            raise ValueError("ModelPolicyJax: ema_params (or params) must be provided for model.apply.")
        variables = self._unwrap_params(variables)

        # sample
        samples = self._jit_apply(variables, subkey, conditions)

        trajectories = samples.trajectories
        values = getattr(samples, "values", jnp.zeros((batch_size,), dtype=jnp.float32))

        # postprocess
        actions = trajectories[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(actions, "actions")
        action = jnp.asarray(actions[0, 0])

        normed_obs = trajectories[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_obs, "observations")

        return action, Trajectories(actions, observations, values)

    def _format_conditions(self, conditions, batch_size):
        # normalize observations
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            "observations",
        )
        # ensure JAX arrays
        conditions = jax.tree_util.tree_map(jnp.asarray, conditions)
        # repeat across batch
        conditions = utils.apply_dict(
            lambda x: einops.repeat(x, "d -> repeat d", repeat=batch_size),
            conditions,
        )
        return conditions
