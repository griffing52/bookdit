#!/usr/bin/env python
"""Rectified-flow rollout.

Euler integration of dx/dt = v(x, t) from noise to data, with an optional observed
prefix pinned to its true noised path (flow inpainting), which is how conditional
rollout is done.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def flow_rollout(model, x_data, steps, observe_frac=None, seed=0):
    """Integrate dx/dt = v(x,t) from t=0 (noise) to t=1 (data), Euler, `steps` steps.

    If observe_frac is set, pin the first `observe_frac` of timesteps to the TRUE noised
    path of x_data (a fixed noise draw) at every step -> conditional forecast of the rest.
    """
    g = torch.Generator(device=x_data.device).manual_seed(seed)
    b, T, D = x_data.shape
    x0 = torch.randn(x_data.shape, generator=g, device=x_data.device)
    x = x0.clone()
    obs = None
    if observe_frac is not None:
        k = int(round(T * observe_frac))
        obs = torch.zeros(1, T, 1, device=x_data.device)
        obs[:, :k] = 1.0                         # observed mask over time
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=x_data.device)
        v = model.forward_features(x, t)
        v = model.velocity_head(model.final_norm(v))
        x = x + dt * v
        if obs is not None:                      # re-pin observed region to its true path
            tnext = (i + 1) * dt
            x = obs * ((1 - tnext) * x0 + tnext * x_data) + (1 - obs) * x
    return x, (None if obs is None else int(obs[0, :, 0].sum().item()))
