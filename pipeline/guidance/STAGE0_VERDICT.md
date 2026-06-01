# Stage 0 verdict — Level-2 catalytic guidance: A_cat fields + decomposed E_cat

**Date:** 2026-05-31
**Scope:** Stage 0 of PI's Level-2 plan — chemistry likelihood E_cat as a sum of
per-fact penalties. No RFD2 modification yet.

## What was built

```
pipeline/guidance/
  a_cat_fields.py        ACatFields — typed-Gaussian A_cat lifted to continuous
                         fields evaluable at world coords. Reads A_cat.json from
                         instantiate_acat (oracle or chem mode).
  e_terms/
    path.py              E_path — substrate-cone exclusion (V_rxn-compatible)
    contact.py           E_contact — typed Gaussian reward (winner-take-all)
    avoid.py             E_avoid — donor poisoning + extra steric clash
    anchor.py            E_anchor — carried anchor reward (no-op until carried)
    seq_chem.py          E_seq_chem — residue-identity penalty (metal-class aware)
    site.py              E_site — continuous V_chem (categorical -> soft)
  e_cat.py               composite E_cat = sum_t lambda_t * E_t
                         + lambda schedules (coarse/mid/fine for SMC annealing)
  damaged_controls.py    10 damage variants (rotated path 90/180, inverted face,
                         wrong hapticity, blocked open site, contact rotated/
                         shuffled, null A_cat)
  discriminativity.py    score one protein under all damage variants; report
                         E_cat with per-term breakdown
```

All 6 E_cat terms passed term-by-term parameter approval (apex_back=0.0,
sigma=0.5 for E_path; winner_take_all + strict types for E_contact;
metal-agnostic donor strengths for E_avoid; closest-sidechain-heavy distance
for E_seq_chem; w_metal=5/w_extra_donor=1.5 for E_site).

## Discriminativity (real protein vs damaged A_cat)

For each target T, score the natural PDB(T) under (real A_cat for T) and 9
damaged variants. Lower E_cat = better catalytic match.

```
TARGET   MODE     REAL E_cat    n_dmg_worse/total     comments
3ZP9     oracle   -3.971        9/9                   full signal across all 6 terms
3ZP9     chem     +0.027        4/9                   E_path/E_avoid fire; E_contact (chem) weak
3WJC     oracle   +0.946        6/9                   no A_path; E_site+E_contact carry signal
5OD5     oracle  +10.854        6/9                   template FP: HIS227 trips E_seq_chem + E_extra_donor
```

## What this means

The PI's diagnosis is confirmed at the term level:

- **E_path** (substrate-cone exclusion): strongly discriminative. Real axis gives
  30x lower E_path than damaged axes. Reliable for any target with an explicit
  open-site atom (synthesized hydride or labile leg).

- **E_avoid** (donor poisoning + steric): strongly discriminative for any
  synthesized poison. CYS at 2A from soft metal: penalty 2.0; HIS: 1.5; ASP: 1.0;
  hierarchy matches textbook donor strength.

- **E_site** (continuous V_chem): categorical separation for wrong metal (+5),
  missing eta5 (+2.5), extra donor (+1.5), wrong hapticity (+3).

- **E_contact** in oracle mode: perfect match (-1 per Gaussian); winner-take-all
  prevents pocket-size bias.

- **E_contact in chem mode (cofactor-only): WEAK.** This is the same finding from
  the heme-transplant test - chemistry-only contact Gaussians don't encode the
  proximal/distal asymmetry needed to recover the natural host. Confirmed at the
  scorer level, not just retrieval level.

- **E_seq_chem** and **E_anchor**: machinery in place but signal modest on
  current data. Need carried anchors populated (5OD5 HIS227, etc.) and richer
  chem-mode rules.

## Known limitations / refinements for Stage 1+

1. **A_path absent for non-hydride targets** (3WJC, 5L8D, 5OD5 in current chem
   instantiator). Fix: extend instantiate_acat to detect open-site from labile
   leg drop OR template's open_site_required flag, not just hydride presence.

2. **Template-specific exemptions** for cp_star_ir_iii_his_ath (5OD5): HIS at
   coord shell is part of the cofactor, not poison. Both E_seq_chem (rule 1
   already has his_exempt_local hook) and E_site (E_extra_donor) need to know
   when to exempt. Trigger: A_cat.A_anchor.carried with allowed_residues=["HIS"].

3. **E_contact chem-mode field** is too sparse. The PI's missing-channels list
   names this: proximal/distal handedness, sequence-side chemistry, electrostatic
   preorganization, dynamics. These need to land BEFORE Stage 2 if guidance is
   to recover the natural host.

## What's READY for Stage 1+

- The likelihood is decomposed, weighted, and tested.
- Sign conventions are consistent (lower = better).
- lambda_t schedules defined (coarse/mid/fine).
- Damaged controls generator is reusable for any future test.

## What's NOT ready

- Stage 1 (outer-loop SMC via guidepost resampling) — implementation pending.
- Stage 2 (RFD2 surgery + in-denoiser particle resampling) — implementation
  pending. Requires sandbox modification.
- Stage 2c (real vs damaged A_cat in-denoiser comparison) — depends on Stage 2.

## Locked parameter card

```
E_path:      apex_back=0.0, sigma_perp=sigma_par=0.5 A, w_bb=w_sc=1.0
E_contact:   winner_take_all, strict type matching, sigma_floor=0.5 A
E_avoid:     w_donor=w_steric=1.0, avoid_radius=2.0 sigma=0.5
             donor strengths: CYS 2.0 / MET 1.2 / HIS 1.5 / ASP/GLU 1.0 / ...
             steric: w_backbone=0.5, w_sidechain=1.0
E_anchor:    sigma=0.8 A, w=1.0 per anchor, closest donor atom selection
E_seq_chem:  CYS=1.0, MET=HIS=0.5 (soft-metal); LYS=ARG=0.5 (hard); cone-charge=0.3
             cationic-TS-stab reward 0.3 for ASP/GLU past-cone-exit
             closest-sidechain-heavy distance; linear falloff
E_site:      w_metal=5.0, w_cn=0.5/count, w_dist=1.0/A, w_hap_missing=2.0,
             w_hap_forbidden=3.0, w_active_missing=1.5, w_extra_donor=1.5
Lambdas:     all 1.0 (default schedule); coarse/mid/fine schedules in e_cat.py
```
