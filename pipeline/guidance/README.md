# Level-2 Catalytic Guidance for OrganoEnzymeGen

PI's plan implemented: chemistry as a separate likelihood, RFD2 as a protein
prior, posterior sampling via SMC.

```
p(P | M_protein, A_cat, S_chem) ~ p_RFD2(P | M_protein) * exp(-E_cat(P; A_cat, S_chem))
```

## Layout

```
pipeline/guidance/
  a_cat_fields.py              ACatFields: typed-Gaussian A_cat lifted to continuous fields
  e_terms/
    path.py                    E_path     (substrate-cone exclusion)
    contact.py                 E_contact  (typed Gaussian reward, winner-take-all)
    avoid.py                   E_avoid    (donor poisoning + extra steric)
    anchor.py                  E_anchor   (carried-anchor reward, off by default)
    seq_chem.py                E_seq_chem (residue-identity penalty, metal-class aware)
    site.py                    E_site     (continuous V_chem + extra-donor)
  e_cat.py                     composite + lambda schedules (coarse/mid/fine)
  damaged_controls.py          10-variant damage generator
  discriminativity.py          full-E_cat probe (real vs damaged A_cat)
  STAGE0_VERDICT.md            Stage 0 result + locked params

  Stage 1 (outer-loop SMC; no RFD2 modification):
    jitter_motif.py            per-particle motif perturbation
    stage1_outer_loop_smc.py   init / score-wave / resample / harvest

  Stage 2 (in-denoiser SMC; requires RFD2 surgery):
    stage2_smc_core.py         generic SMC particle filter
    stage2_rfd2_hook_SPEC.md   cluster-side hook interface contract
    stage2_rfd2_hook_template.py  skeleton (fill in against RFD2 source)

  _test_*.py                   sanity / discriminativity tests
```

## Stage 0 result (locked)

Real-vs-damaged discriminativity validated term-by-term. All 6 E_cat terms
have approved parameters (see STAGE0_VERDICT.md). E_path, E_avoid, E_site
strongly discriminative; E_contact in chem mode weak (matches PI's report
9.5 heme-transplant finding at the scorer level).

Locked parameters (overridable per-term at call time):

```python
ACatFields(a_cat, path_sigma_perp=0.5, path_sigma_par=0.5,
                  path_apex_offset_back=0.0,
                  contact_sigma_floor=0.5,
                  avoid_radius_around_metal=2.0, avoid_sigma=0.5)

e_cat lambdas: all 1.0 (override per stage; see SCHEDULE_{COARSE,MID,FINE})
```

## Stage 1 usage (cluster-runnable)

```bash
# 1. Set up the workdir + wave-0 particles (K=8, N=50 designs/particle, 3 waves)
python pipeline/guidance/stage1_outer_loop_smc.py init \
    --target 3ZP9 \
    --base-motif pipeline/compiled/3ZP9/motif.pdb \
    --a-cat pipeline/compiled/3ZP9/A_cat.json \
    --workdir runs/smc/3ZP9 --k 8 --n 50 --waves 3 \
    --sigma-init 0.5 --sigma-div 0.3 --lam 1.0

# 2. Submit each particle's RFD2 job (sbatch files pre-rendered by init)
for k in 000 001 002 003 004 005 006 007; do
    sbatch runs/smc/3ZP9/wave_000/particle_$k/submit_rfd2.sbatch
done

# 3. After all particles finish (uses standard squeue/sacct monitoring):
python pipeline/guidance/stage1_outer_loop_smc.py score-wave --workdir runs/smc/3ZP9 --wave 0
python pipeline/guidance/stage1_outer_loop_smc.py resample   --workdir runs/smc/3ZP9 --wave 0

# 4. Repeat for waves 1, 2 (submit -> score-wave -> resample)
# 5. Harvest top designs:
python pipeline/guidance/stage1_outer_loop_smc.py harvest --workdir runs/smc/3ZP9 --top 20
```

Compute cost (rough): K * N * waves = 8 * 50 * 3 = 1200 designs ~ same as 12
baseline runs of 100 designs. ~1 hour per wave on V100s.

## Stage 2 usage (requires RFD2 surgery)

See `stage2_rfd2_hook_SPEC.md` for the full hook interface contract.

```bash
# After implementing the cluster-side hook in stage2_rfd2_hook_template.py,
# vanilla preservation test (lambda_max=0 must reproduce baseline):
python stage2_rfd2_hook_template.py \
    --motif pipeline/compiled/3ZP9/motif.pdb \
    --a-cat pipeline/compiled/3ZP9/A_cat.json \
    --ligand LIG --contigs "..." --contig-atoms "..." \
    --ckpt $REPO_DIR/rf_diffusion/model_weights/RFD_173.pt \
    --out-dir runs/smc2/3ZP9_vanilla \
    -K 8 --checkpoint-every 10 --lambda-max 0.0 --seed 42

# Compare V_chem / V_rxn pass rates to baseline 100-design ensemble.
# If they match within tolerance, proceed:

python stage2_rfd2_hook_template.py \
    ... --lambda-max 1.0 --tiered-lambdas \
    --out-dir runs/smc2/3ZP9_guided

# Stage 2c (real vs damaged A_cat):
# Pre-generate damaged A_cat JSONs:
python pipeline/guidance/damaged_controls.py pipeline/compiled/3ZP9/A_cat.json \
    --out-dir pipeline/compiled/3ZP9/damaged/
# Run SMC for each variant:
for v in real rotated_path_180 inverted_face wrong_hapticity; do
    python stage2_rfd2_hook_template.py ... --a-cat pipeline/compiled/3ZP9/damaged/$v.json \
        --out-dir runs/smc2/3ZP9_dmg/$v
done
# Cross-evaluate finals against REAL A_cat:
for v in ...; do
    python pipeline/v_chem.py runs/smc2/3ZP9_dmg/$v
    python pipeline/v_rxn.py runs/smc2/3ZP9_dmg/$v
done
```

## Risks I flagged during build

1. **E_contact chem-mode is weak** (4/9 damages discriminate vs 9/9 in oracle).
   Chemistry-only Gaussians don't encode proximal/distal asymmetry. Until this
   improves, Stage 2 guidance under chem-mode A_cat may produce shape-coherent
   but host-distinct designs (the same Stage 9.5 heme-transplant failure mode
   one layer deeper).

2. **A_path absent for non-hydride targets** (3WJC, 5L8D, current 5OD5). The
   instantiator extension needed: detect open site from labile-leg drop OR
   from V_chem template's `open_site_required: false` + His coord rule.

3. **5OD5 false positives** in E_seq_chem (HIS227 trips soft-metal-poison
   rule) and E_site (HIS227 counts as extra protein donor). Both terms have
   `carried-anchor exemption` hooks; need carried anchors populated for 5OD5.

4. **Mock denoiser bias-to-clean prevents Stage 2c discriminativity in test
   harness.** Stage 2c is fundamentally a cluster experiment with the real
   RFD2 protein-fold prior. Mock validates SMC mechanics only.

5. **RFD2 hook completeness**: the `atoms()` conversion is the highest-risk
   piece. Must emit sequence-assigned backbone + cofactor in WORLD coords.
   E_contact's sidechain-centroid logic falls back to CA if sidechains are
   missing (E_path, E_avoid, E_site all work backbone-only).

## Pre-Stage-2-deployment checklist

- [ ] Implement RFD2DenoiserHook against current sandbox RFD2 source.
- [ ] Vanilla preservation test (lambda_max=0 == baseline within tolerance).
- [ ] Memory profile (K=8, K=16 viability check).
- [ ] Verify x_hat_0 quality at checkpoint schedule (skip first few if too noisy).
- [ ] Resolve E_contact chem-mode weakness OR commit to oracle A_cat for Stage 2c.
- [ ] Add carried anchors to A_cat for 5OD5 (template-specific enrichment).
- [ ] Add A_path for non-hydride targets (extend instantiate_acat).
