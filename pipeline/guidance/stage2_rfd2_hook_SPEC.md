# Stage 2 RFD2 Hook — Specification

**Purpose**: enable in-denoiser SMC guidance for RFD2 (Stage 2 of the Level-2
posterior-sampling plan). The hook is the bridge between RFD2's internal
denoising state and the chemistry-aware likelihood E_cat.

**Where this lives**: inside the RFD2 sandbox on the cluster. Must import
RFD2 modules; the Windows-side `stage2_smc_core.py` does not.

## Required interface

```python
class RFD2DenoiserHook:
    """User implements this. Wraps RFD2's inference loop."""

    @property
    def total_steps(self) -> int:
        """T — total number of denoising steps in this RFD2 schedule.
        Typically 50-100 for RFD2."""
        ...

    def init_x(self) -> Any:
        """Return one independent initial noise sample (could be a tensor
        dict, torch state, or any python object the hook understands).
        Called K times for K particles."""
        ...

    def step(self, x_t, t: int) -> tuple:
        """Advance from x_t to x_{t-1} for one particle, returning:
            (x_{t-1}, x_hat_0)
        where x_hat_0 is the model's CURRENT estimate of the clean structure.
        For RFD2 (DDPM-style), x_hat_0 is the predicted x_0 from the noise
        prediction: x_hat_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t).
        See RFD2's run_inference loop for the exact extraction."""
        ...

    def atoms(self, x) -> list:
        """Convert a state (typically x_hat_0) to a list of atom dicts
        compatible with the E_cat scorer:
            [{"record": "ATOM"|"HETATM", "name": "...", "element": "...",
              "resname": "...", "chain": "...", "resseq": int,
              "x": float, "y": float, "z": float}, ...]
        For RFD2, x_hat_0 typically contains backbone frames (N, CA, C, O
        coordinates per residue) and a sequence prediction. The hook must:
          - Use the predicted/assigned sequence to set resname.
          - Emit all backbone atoms (N, CA, C, O).
          - Emit the rigid cofactor atoms unchanged (they are fixed).
          - If sidechains are predicted (LigandMPNN-style), include them too;
            otherwise emit only backbone (E_cat handles missing sidechains
            gracefully — E_path uses backbone too, E_contact uses sidechain
            centroid which falls back to CA for missing sidechains)."""
        ...

    def write(self, x, path: str) -> None:
        """Write a final particle to PDB at `path`. Called for the K final
        particles after SMC completes."""
        ...
```

## Cluster-side implementation sketch

```python
# pipeline/guidance/stage2_rfd2_hook_cluster.py  (cluster-only)

import sys, os, torch
sys.path.insert(0, os.environ["REPO_DIR"])     # RFD2 repo root
from rf_diffusion.run_inference import build_sampler, predict_clean
# NOTE: actual import paths vary by RFD2 version; verify with the live repo.


class RFD2DenoiserHook:
    def __init__(self, motif_pdb, ligand, contigs, contig_atoms, ckpt_path):
        self.sampler, self.context = build_sampler(
            motif_pdb=motif_pdb, ligand=ligand,
            contigs=contigs, contig_atoms=contig_atoms,
            ckpt_path=ckpt_path,
        )

    @property
    def total_steps(self):
        return self.sampler.T

    def init_x(self):
        return self.sampler.sample_initial_noise(self.context)

    def step(self, x_t, t):
        # 1. model forward
        eps_pred, seq_logits = self.sampler.model(x_t, t, self.context)
        # 2. clean estimate x_hat_0
        x_hat_0 = self.sampler.predict_x0(x_t, eps_pred, t)
        # 3. step to x_{t-1}  (use sampler's standard transition)
        x_next = self.sampler.step(x_t, eps_pred, t)
        # bundle sequence info into x_hat_0 if needed
        x_hat_0.seq = seq_logits.argmax(-1)
        return x_next, x_hat_0

    def atoms(self, x):
        # RFD2's x has .frames (T,N,4,3 or similar) and .seq.
        # Convert backbone frames to N/CA/C/O atom coords, decode resname
        # from .seq via AA_ALPHABET, include cofactor unchanged.
        out = []
        # ... see rf_diffusion.utils.frames_to_atoms for the canonical conversion
        return out

    def write(self, x, path):
        atoms = self.atoms(x)
        # ... standard PDB writer over `atoms`
```

## Calling from the SMC driver

```python
# Cluster-side driver: pipeline/guidance/stage2_run_smc.py
from stage2_smc_core import run_smc
from stage2_rfd2_hook_cluster import RFD2DenoiserHook
from a_cat_fields import load
from e_cat import e_cat

fields = load("path/to/A_cat.json")
hook = RFD2DenoiserHook(motif_pdb=..., ligand="LIG", contigs=..., contig_atoms=..., ckpt_path=...)
result = run_smc(hook, fields, e_cat,
                 K=16, checkpoint_every=10, lambda_max=1.0,
                 schedule="linear", ess_threshold_frac=0.5, seed=42, verbose=True)
for k, x in enumerate(result["particles"]):
    hook.write(x, f"smc_p{k:03d}.pdb")
```

## Validation requirements (Stage 2a milestone)

Before running real SMC, demonstrate **vanilla preservation**:
  - Run with K=4, `lambda_max=0` (or all `tiered_lambdas` weights = 0).
  - Expect: outputs distributionally identical to un-guided RFD2 sampling.
  - Check: Layer-1 "preserved" rates, V_chem PASS rates, V_rxn pass rates
    match within tolerance against 100-design baseline.

If vanilla preservation passes, proceed to Stage 2b (run with `lambda_max > 0`).

## Risks / gotchas

1. **x_hat_0 quality at high noise**: at t close to T, the model's clean
   estimate is essentially random. Scoring it with E_cat would penalize all
   particles equally → no useful gradient. Use `checkpoint_every` to skip
   early-noise checkpoints (or use a sigmoid/tiered lambda schedule).

2. **Particle collapse**: aggressive resampling can collapse all particles
   to a single parent. Monitor ESS; raise `ess_threshold_frac` or lower
   `lambda_max` if it crashes early.

3. **Memory**: K particles × full RFD2 state may exceed GPU memory. Either:
   - Process particles serially in a loop (slower but no memory pressure).
   - Batch K=2 or 4 if memory permits.

4. **Stochasticity reproducibility**: RFD2's `inference.deterministic=True`
   may need to be `False` for K independent noise samples, but the SMC
   trajectory should be reproducible given `seed`.

5. **Hook completeness**: the `atoms` conversion is the highest-risk piece.
   E_cat needs WORLD coordinates with correct sequence labels; if the hook
   emits backbone-only atoms, E_contact's sidechain-centroid logic falls
   back to CA — works but loses some discrimination.

## What this is NOT

- Not gradient-based guidance (i.e., not classifier-guided diffusion in the
  ∇log-likelihood sense). SMC resampling is a particle filter, not gradient
  flow. The PI's framing also recommended SMC over gradient hacking.
- Not LigandMPNN-aware (yet). Sequence-level guidance via E_seq_chem can
  bias the sequence prediction step but this hook does not yet expose that
  bias. Add it once Stage 2b is validated.
