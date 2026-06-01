#!/usr/bin/env python3
"""_test_stage2_real_vs_damaged.py - Stage 2c probe with mock denoiser.

Run SMC under (real A_cat) and several (damaged A_cat) variants using the
same mock noise schedule. Expected:
  - Real A_cat: particles converge toward clean target -> low final E_cat
  - Damaged A_cat: SMC pulls particles AWAY from clean target -> finals look
    like a different optimum (low E_cat under damaged, high under real)

This validates the SMC algorithm + E_cat scorer are doing chemistry-faithful
work BEFORE we burn cluster compute on real RFD2 SMC.
"""
from __future__ import annotations
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_cat import e_cat
from guidance.stage2_smc_core import run_smc
from guidance.damaged_controls import ALL_DAMAGES
from guidance._score_pdb import parse_pdb
from guidance._test_stage2_smc import MockDenoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    ap.add_argument("-K", type=int, default=12)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--sigma-init", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields_real = ACatFields(a)
    clean_full = parse_pdb(args.pdb)
    nearby = [at for at in clean_full
              if (lambda pl: (pl[0]**2+pl[1]**2+pl[2]**2)**0.5 <= 14.0)(fields_real.to_local((at["x"], at["y"], at["z"])))
              or at.get("resname") in ("LIG", "ORI")]
    print(f"# Clean target n_atoms={len(nearby)} (within 14 A of cofactor)")
    print(f"# Clean target E_cat (real A_cat) = {e_cat(nearby, fields_real):.3f}")

    variant_names = ["real", "rotated_path_180", "inverted_face", "wrong_hapticity"]
    fmt = "{:<20} {:>10} {:>10} {:>10} {:>14}"
    print()
    print(fmt.format("A_cat used in SMC", "final_min", "final_mean", "final_max", "E_cat(clean|A)"))
    print(fmt.format("-"*20, "-"*10, "-"*10, "-"*10, "-"*14))
    for vname in variant_names:
        a_var = ALL_DAMAGES[vname](json.loads(json.dumps(a)))
        fields_var = ACatFields(a_var)
        clean_E = e_cat(nearby, fields_var)
        rng = random.Random(args.seed)
        hook = MockDenoiser(nearby, sigma_init=args.sigma_init, total_steps=args.T, rng=rng)
        result = run_smc(hook, fields_var, e_cat,
                         K=args.K, checkpoint_every=args.checkpoint_every,
                         lambda_max=args.lambda_max, schedule="linear",
                         ess_threshold_frac=0.5, seed=args.seed, verbose=False)
        # score finals under the VARIANT A_cat (what SMC was optimizing)
        finals_under_variant = [e_cat(hook.atoms(p), fields_var) for p in result["particles"]]
        finals_under_variant.sort()
        print(fmt.format(vname, f"{min(finals_under_variant):.3f}",
                                  f"{sum(finals_under_variant)/len(finals_under_variant):.3f}",
                                  f"{max(finals_under_variant):.3f}",
                                  f"{clean_E:.3f}"))

    # Also: cross-evaluate - run SMC under damaged A_cat, then score FINALS
    # against REAL A_cat. If damaged SMC was effective, finals should look
    # GOOD under damaged but BAD under real.
    print()
    print("Cross-eval: SMC under damaged A_cat, finals scored against REAL A_cat")
    print(fmt.format("A_cat in SMC", "min(real)", "mean(real)", "max(real)", "E_cat(clean|R)"))
    print(fmt.format("-"*20, "-"*10, "-"*10, "-"*10, "-"*14))
    for vname in variant_names:
        a_var = ALL_DAMAGES[vname](json.loads(json.dumps(a)))
        fields_var = ACatFields(a_var)
        rng = random.Random(args.seed)
        hook = MockDenoiser(nearby, sigma_init=args.sigma_init, total_steps=args.T, rng=rng)
        result = run_smc(hook, fields_var, e_cat,
                         K=args.K, checkpoint_every=args.checkpoint_every,
                         lambda_max=args.lambda_max, schedule="linear",
                         ess_threshold_frac=0.5, seed=args.seed, verbose=False)
        # score finals against REAL A_cat
        finals_under_real = [e_cat(hook.atoms(p), fields_real) for p in result["particles"]]
        finals_under_real.sort()
        clean_real = e_cat(nearby, fields_real)
        print(fmt.format(vname, f"{min(finals_under_real):.3f}",
                                  f"{sum(finals_under_real)/len(finals_under_real):.3f}",
                                  f"{max(finals_under_real):.3f}",
                                  f"{clean_real:.3f}"))


if __name__ == "__main__":
    main()
