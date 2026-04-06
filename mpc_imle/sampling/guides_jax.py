from dataclasses import dataclass
from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _to_B(y: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize value outputs to shape [B].

    Accepts:
      - [B]
      - [B, 1]

    Returns:
      - [B]

    Raises if anything else (e.g. [B, H], [B, H, 1]).
    """
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[-1] == 1:
        return y[:, 0]
    raise ValueError(f"Value guide expected output [B] or [B,1], got {y.shape}")


def _normalize_t(t: Optional[Any], batch_size: int) -> jnp.ndarray:
    """
    Normalize timestep input to int32 array of shape [B].

    Handles:
      - None
      - Python int / float
      - scalar JAX array
      - [B] JAX array
    """
    if t is None:
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    if isinstance(t, (int, float)):
        return jnp.full((batch_size,), int(t), dtype=jnp.int32)

    t = jnp.asarray(t, dtype=jnp.int32)
    if t.ndim == 0:
        return jnp.full((batch_size,), t, dtype=jnp.int32)

    return t


# ---------------------------------------------------------------------
# Base value guide
# ---------------------------------------------------------------------

@dataclass
class ValueGuideJax:
    apply_fn: Any   # usually model.apply
    params: Any     # variables["params"]

    def __call__(self, x: jnp.ndarray, cond: Any, t: Any) -> jnp.ndarray:
        """
        Args:
          x:    [B, H, D]
          cond: conditioning (dict or tensor)
          t:    scalar / [B] timestep

        Returns:
          values: [B]
        """
        t0 = _normalize_t(t, x.shape[0])
        y = self.apply_fn({"params": self.params}, x, cond, t0)
        return _to_B(y)

    def value_and_gradients(
        self, x: jnp.ndarray, cond: Any, t: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
          values: [B]
          grad:   same shape as x
        """
        def f(x_in):
            vals = self(x_in, cond, t)  # [B]
            return jnp.sum(vals)        # scalar

        grad = jax.grad(f)(x)
        values = self(x, cond, t)
        return values, grad


# ---------------------------------------------------------------------
# IMLE value guide (t ignored / fixed)
# ---------------------------------------------------------------------

@dataclass
class IMLEValueGuideJax(ValueGuideJax):
    """
    IMLE-style value guide:
    - ignores provided t
    - always evaluates value at t = 0
    """

    def __call__(self, x: jnp.ndarray, cond: Any, t: Optional[Any] = None) -> jnp.ndarray:
        t0 = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        y = self.apply_fn({"params": self.params}, x, cond, t0)
        return _to_B(y)

    def value_and_gradients(
        self, x: jnp.ndarray, cond: Any, t: Optional[Any] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def f(x_in):
            vals = self(x_in, cond, None)
            return jnp.sum(vals)

        grad = jax.grad(f)(x)
        values = self(x, cond, None)
        return values, grad
