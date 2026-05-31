#!/usr/bin/env python3
"""stage2_smc_core.py - in-denoiser sequential Monte Carlo for RFD2 + E_cat.

This is the GENERIC SMC algorithm. It does not import RFD2. The caller
provides a `DenoiserHook` (cluster-side, inside the RFD2 sandbox) that
exposes:

    hook.total_steps   : int T
    hook.init_x()      : initial noise state for one particle
    hook.step(x, t)    : (x_next, x_hat_0)  -- next denoising step + clean estimate
    hook.atoms(x)      : list of atom dicts compatible with E_cat scorer
    hook.write(x, path): serialize a particle's state to a PDB (final output only)

The SMC algorithm:
  1. Initialize K particles {x^k_T} = K independent noise samples.
  2. For t = T, T-1, ..., 1:
        a. Step every particle: (x^k_{t-1}, x_hat0^k) = hook.step(x^k_t, t).
        b. At every R-th step (R = checkpoint_every), or if t == 1:
              - Evaluate E_cat(hook.atoms(x_hat0^k)) for each k.
              - Compute weight w_k = exp(-lambda(t) * E_cat^k).
              - Effective sample size ESS = 1 / sum(w_k^2).
              - If ESS < threshold * K: resample particles with replacement.
              - Else: keep particles, multiply weights into running log_w.
  3. Return final particles + score history.

lambda(t) schedule controls how aggressive guidance is at each noise level.
Default schedule is monotone in 1 - t/T (no guidance at high noise; full
guidance at low noise).

This module is unit-testable with a MOCK denoiser that returns deterministic
"states" (e.g., perturbed copies of a real motif). The real RFD2 hook lives
in stage2_rfd2_hook_template.py (cluster-side; needs RFD2 source).
"""
from __future__ import annotations
import math, random
from typing import Callable, List, Tuple


def lambda_schedule_linear(t: int, T: int, lambda_max: float = 1.0) -> float:
    """No guidance at t=T (full noise), full guidance at t=1 (clean).
    Linear in (1 - t/T)."""
    return lambda_max * max(0.0, 1.0 - (t / T))


def lambda_schedule_sigmoid(t: int, T: int, lambda_max: float = 1.0,
                            midpoint_frac: float = 0.5, steepness: float = 10.0) -> float:
    """Smooth onset of guidance around midpoint_frac of denoising."""
    progress = 1.0 - (t / T)
    return lambda_max / (1.0 + math.exp(-steepness * (progress - midpoint_frac)))


def lambda_schedule_tiered(t: int, T: int) -> dict:
    """Per-term schedule: which terms are active at this noise level.
    Returns a dict {term_name: weight}. Matches e_cat.SCHEDULE_{COARSE,MID,FINE}."""
    progress = 1.0 - (t / T)
    if progress < 0.33:    # high noise: coarse only
        return {"path": 1.0, "contact": 0.0, "avoid": 0.0,
                "anchor": 0.0, "seq_chem": 0.0, "site": 0.0}
    elif progress < 0.66:  # mid: + pocket topology
        return {"path": 1.0, "contact": 1.0, "avoid": 1.0,
                "anchor": 0.0, "seq_chem": 0.0, "site": 0.0}
    else:                   # low noise: full
        return {"path": 1.0, "contact": 1.0, "avoid": 1.0,
                "anchor": 1.0, "seq_chem": 1.0, "site": 1.0}


def _resample_systematic(weights_norm: List[float], rng: random.Random) -> List[int]:
    """Systematic resampling: K positions evenly spaced + random offset.
    Lower variance than multinomial. Returns K parent indices."""
    K = len(weights_norm)
    u0 = rng.random() / K
    edges = []
    s = 0.0
    for w in weights_norm:
        s += w
        edges.append(s)
    parents = []
    idx = 0
    for j in range(K):
        u = u0 + j / K
        while idx < K - 1 and u > edges[idx]:
            idx += 1
        parents.append(idx)
    return parents


def run_smc(hook, fields, e_cat_fn, *,
            K: int = 16,
            checkpoint_every: int = 10,
            lambda_max: float = 1.0,
            schedule: str = "linear",
            ess_threshold_frac: float = 0.5,
            tiered_lambdas: bool = False,
            seed: int = 0,
            verbose: bool = True):
    """Run in-denoiser SMC. Returns dict with final particles and trajectory log.

    Args:
      hook: DenoiserHook (see module docstring).
      fields: ACatFields - the catalytic likelihood.
      e_cat_fn: callable(atoms_list, fields[, lambdas=...]) -> scalar
                If tiered_lambdas, expects to accept `lambdas` kwarg.
      K: number of particles.
      checkpoint_every: every R denoising steps, resample.
      lambda_max: max lambda in schedule (overall guidance strength).
      schedule: 'linear' | 'sigmoid'.
      ess_threshold_frac: resample only when ESS < frac * K (keeps diversity).
      tiered_lambdas: if True, use lambda_schedule_tiered for per-term weights.
                      else use scalar lambda for the unweighted E_cat sum.
      seed: RNG seed.
      verbose: if True, log per-checkpoint status.

    Returns: { "particles": [...], "log": [...], "weights": [...] }
    """
    rng = random.Random(seed)
    T = hook.total_steps
    particles = [hook.init_x() for _ in range(K)]
    log_weights = [0.0] * K     # log of un-normalized importance weight
    history = []

    for step_idx, t in enumerate(range(T, 0, -1)):
        # 1. step every particle
        next_particles = []; x_hat0_list = []
        for x in particles:
            xn, x0 = hook.step(x, t)
            next_particles.append(xn); x_hat0_list.append(x0)

        # 2. check if this is a checkpoint
        is_checkpoint = (step_idx % checkpoint_every == 0) or (t == 1)
        if is_checkpoint:
            # determine lambda(s)
            if tiered_lambdas:
                lam_dict = lambda_schedule_tiered(t, T)
            else:
                lam_scalar = (lambda_schedule_linear(t, T, lambda_max)
                              if schedule == "linear"
                              else lambda_schedule_sigmoid(t, T, lambda_max))
            # 3. score each particle's x_hat_0
            E_list = []
            for x0 in x_hat0_list:
                atoms = hook.atoms(x0)
                if tiered_lambdas:
                    E = e_cat_fn(atoms, fields, lambdas=lam_dict)
                else:
                    E = e_cat_fn(atoms, fields)
                E_list.append(E)
            # 4. log-weight update
            if tiered_lambdas:
                delta_logw = [-E for E in E_list]   # lambdas already in per-term weights
            else:
                delta_logw = [-lam_scalar * E for E in E_list]
            log_weights = [lw + dlw for lw, dlw in zip(log_weights, delta_logw)]
            # normalize for ESS
            offset = max(log_weights)
            wnorm = [math.exp(lw - offset) for lw in log_weights]
            s = sum(wnorm)
            wnorm = [w / s for w in wnorm]
            ess = 1.0 / sum(w*w for w in wnorm)
            do_resample = ess < ess_threshold_frac * K
            entry = {"step": step_idx, "t": t,
                     "lambda": (lam_dict if tiered_lambdas else lam_scalar),
                     "E_min": round(min(E_list), 4),
                     "E_mean": round(sum(E_list) / len(E_list), 4),
                     "E_max": round(max(E_list), 4),
                     "ESS": round(ess, 3), "resampled": False}
            if do_resample:
                parents = _resample_systematic(wnorm, rng)
                next_particles = [next_particles[i] for i in parents]
                log_weights = [0.0] * K   # reset
                entry["resampled"] = True
                entry["parents"] = parents
            history.append(entry)
            if verbose:
                print(f"  [t={t:3d}] E in [{min(E_list):8.3f}, {max(E_list):8.3f}] "
                      f"ESS={ess:.2f}/{K}  {'RESAMPLED' if do_resample else 'kept'}")

        particles = next_particles

    return {"particles": particles, "history": history, "final_log_weights": log_weights}
