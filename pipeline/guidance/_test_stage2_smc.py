#!/usr/bin/env python3
"""_test_stage2_smc.py - validate stage2_smc_core with a MOCK denoiser.

Mock denoiser semantics:
  state = list of atom dicts (real 3ZP9 protein + decaying gaussian noise on
          every coordinate)
  init_x: high-noise (sigma_init A) perturbation of real protein
  step:   reduce noise scale linearly with t/T
  x_hat_0: state with noise scaled by (t/T)^2 (model's clean estimate)

Expected behavior:
  - Without guidance (lambda=0): final E_cat = init E_cat on average (denoising
    only removes the noise we added — net zero signal).
  - With guidance (lambda>0):  particles selected toward lower E_cat;
    final ensemble has LOWER mean E_cat than the no-guidance baseline.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_cat import e_cat
from guidance.stage2_smc_core import run_smc
from guidance._score_pdb import parse_pdb


class MockDenoiser:
    """Decaying-noise sampler over a real protein conformation.
    Each particle has its OWN random direction, so two particles are
    distinct, but they all approach the same clean target."""

    def __init__(self, clean_atoms, sigma_init=2.0, total_steps=50, rng=None):
        self.clean = clean_atoms
        self.sigma_init = sigma_init
        self.T = total_steps
        self.rng = rng or random.Random(0)

    @property
    def total_steps(self):
        return self.T

    def _sigma_at(self, t):
        # linear decay from sigma_init at t=T to 0.05 at t=1
        frac = t / self.T
        return 0.05 + (self.sigma_init - 0.05) * frac

    def init_x(self):
        sigma = self._sigma_at(self.T)
        return self._perturb(self.clean, sigma)

    def step(self, x, t):
        # x_{t-1}: a fresh perturbation with smaller sigma + a hint of bias toward clean
        sigma_next = self._sigma_at(t - 1)
        # interpolate toward clean by (1 - t/T)
        alpha = 1.0 - (t / self.T)
        x_next = []
        for a, c in zip(x, self.clean):
            nx = (1 - 0.1) * a["x"] + 0.1 * c["x"] + self.rng.gauss(0, sigma_next)
            ny = (1 - 0.1) * a["y"] + 0.1 * c["y"] + self.rng.gauss(0, sigma_next)
            nz = (1 - 0.1) * a["z"] + 0.1 * c["z"] + self.rng.gauss(0, sigma_next)
            an = copy.copy(a); an["x"] = nx; an["y"] = ny; an["z"] = nz
            x_next.append(an)
        # x_hat_0: model's clean estimate = state with noise further reduced by (t/T)^2
        x_hat0 = []
        scale = (t / self.T) ** 2
        for a, c in zip(x_next, self.clean):
            nx = c["x"] + scale * (a["x"] - c["x"])
            ny = c["y"] + scale * (a["y"] - c["y"])
            nz = c["z"] + scale * (a["z"] - c["z"])
            an = copy.copy(a); an["x"] = nx; an["y"] = ny; an["z"] = nz
            x_hat0.append(an)
        return x_next, x_hat0

    def atoms(self, x):
        return x

    def write(self, x, path):
        with open(path, "w") as f:
            for i, a in enumerate(x, 1):
                rec = a["record"]
                nm = a["name"]; rn = a["resname"]; ch = a.get("chain", "A")
                rs = a.get("resseq", 1)
                f.write(f"{rec:<6}{i:>5} {nm:<4} {rn:<3} {ch}{rs:>4}    "
                        f"{a['x']:>8.3f}{a['y']:>8.3f}{a['z']:>8.3f}"
                        f"{1.0:>6.2f}{0.0:>6.2f}          {a.get('element', nm[0]):>2}\n")
            f.write("END\n")

    def _perturb(self, atoms, sigma):
        out = []
        for a in atoms:
            an = copy.copy(a)
            an["x"] = a["x"] + self.rng.gauss(0, sigma)
            an["y"] = a["y"] + self.rng.gauss(0, sigma)
            an["z"] = a["z"] + self.rng.gauss(0, sigma)
            out.append(an)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--sigma-init", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields = ACatFields(a)
    clean = parse_pdb(args.pdb)
    # for speed: trim to the cofactor's local 12 A neighborhood
    nearby = []
    for at in clean:
        pl = fields.to_local((at["x"], at["y"], at["z"]))
        d = (pl[0]**2 + pl[1]**2 + pl[2]**2) ** 0.5
        if d <= 14.0 or at.get("resname") in ("LIG", "ORI"):
            nearby.append(at)
    print(f"# Clean target n_atoms={len(nearby)} (within 14 A of cofactor)")
    print(f"# K={args.K}  T={args.T}  sigma_init={args.sigma_init}  ckpt_every={args.checkpoint_every}")
    print(f"# E_cat on clean target (sanity): {e_cat(nearby, fields):.4f}")

    for lam in (0.0, args.lambda_max):
        print(f"\n=== SMC with lambda_max={lam} (no_guidance={lam==0}) ===")
        rng = random.Random(args.seed)
        hook = MockDenoiser(nearby, sigma_init=args.sigma_init, total_steps=args.T, rng=rng)
        result = run_smc(hook, fields, e_cat,
                         K=args.K, checkpoint_every=args.checkpoint_every,
                         lambda_max=lam, schedule="linear",
                         ess_threshold_frac=0.5, seed=args.seed, verbose=False)
        # final E_cat distribution
        final_E = [e_cat(hook.atoms(p), fields) for p in result["particles"]]
        final_E.sort()
        print(f"  history (compact):")
        for h in result["history"][::max(1, len(result['history'])//6)][:7]:
            print(f"    t={h['t']:3d}  E[min,mean,max]=[{h['E_min']:7.3f}, "
                  f"{h['E_mean']:7.3f}, {h['E_max']:7.3f}]  ESS={h['ESS']:.2f}  "
                  f"{'RESAMPLED' if h['resampled'] else ''}")
        print(f"  final E_cat: min={min(final_E):.3f}  mean={sum(final_E)/len(final_E):.3f}  max={max(final_E):.3f}")


if __name__ == "__main__":
    main()
