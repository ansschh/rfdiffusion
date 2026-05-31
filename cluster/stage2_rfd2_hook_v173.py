#!/usr/bin/env python3
"""stage2_rfd2_hook_v173.py - RFD2 hook for in-denoiser SMC.

Tested against the v173 API as seen in rf_diffusion/run_inference.py:
    sampler = model_runners.sampler_selector(conf)
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(i_des)
    ts = torch.arange(t_step_input, sampler.inf_conf.final_step - 1, -1)
    for t in ts:
        px0, x_t, seq_t, rfo, extra = sampler.sample_step(
            t, indep, rfo, extra, features_cache)
        indep.xyz = x_t
        ...

For SMC:
  - K particles, each with its own (indep, rfo, extra, features_cache)
  - At each t, advance all K via sample_step (sequentially; K-fold compute)
  - At scheduled checkpoints: score each particle's px0 with E_cat,
    resample particles weighted by exp(-lambda(t) * E_cat)

Must run INSIDE the RFD2 sandbox. Run via:

  cd $RFD2
  apptainer exec --nv -B /resnick --home $RFD2HOME \\
    --env DGLBACKEND=pytorch --env MKL_THREADING_LAYER=GNU \\
    --env MKL_SERVICE_FORCE_INTEL=1 \\
    rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox \\
    python /resnick/scratch/atiwari2/rfdiffusion/cluster/stage2_rfd2_hook_v173.py [args]

VANILLA PRESERVATION GATE: run with --K 1 --lambda-max 0.0 and verify the
output matches a baseline RFD2 design at the same seed.
"""
from __future__ import annotations
import argparse, copy, json, math, os, pickle, random, sys, time
from collections import defaultdict
from typing import List

import torch
import numpy as np

# --- import RFD2 (must run inside the sandbox) -----------------------------
RFD2_DIR = os.environ.get("RFD2_DIR", "/resnick/scratch/atiwari2/RFdiffusion2")
REPO_DIR = os.environ.get("REPO_DIR", "/resnick/scratch/atiwari2/rfdiffusion")
sys.path.insert(0, RFD2_DIR)                                      # rf_diffusion package root
sys.path.insert(0, os.path.join(REPO_DIR, "pipeline"))            # guidance package

import hydra
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig
from icecream import ic
ic.configureOutput(includeContext=True)

from rf_diffusion.inference import model_runners
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.chemical import reinitialize_chemical_data
from rf_diffusion import aa_model
from rf_diffusion import atomize
from rf_diffusion import features as rfd_features
from rf_diffusion.dev import idealize_backbone
from rf_diffusion.idealize import idealize_pose
from rf_diffusion.import_pyrosetta import prepare_pyrosetta
import rf2aa.tensor_util
import rf2aa.util

# Local imports
from guidance.a_cat_fields import load as load_fields
from guidance.e_cat import e_cat as e_cat_fn, DEFAULT_LAMBDAS, SCHEDULE_FINE
from guidance.stage2_smc_core import (
    lambda_schedule_linear, lambda_schedule_sigmoid, lambda_schedule_tiered,
    _resample_systematic,
)

# RFD2 uses the standard 20-AA alphabet; map index -> 3-letter resname
# (verified against rf2aa.chemical.aa_321 / aa_long_short)
AA_THREE = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
            "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]


def aa_idx_to_resname(i: int) -> str:
    return AA_THREE[i] if 0 <= i < len(AA_THREE) else "ALA"


# --- Hydra config build (mirrors run_inference.main) -----------------------

def build_conf(motif_pdb: str, ligand: str, contigs: str, contig_atoms: str,
               ckpt_path: str, output_prefix: str, num_designs: int = 1,
               deterministic: bool = False) -> DictConfig:
    """Build the same Hydra config object run_inference.py would use."""
    cfg_dir = os.path.join(RFD2_DIR, "rf_diffusion/config/inference")
    overrides = [
        f"inference.deterministic={'True' if deterministic else 'False'}",
        f"inference.ckpt_path={ckpt_path}",
        f"inference.input_pdb={motif_pdb}",
        f"inference.ligand={ligand}",
        f'contigmap.contigs={contigs}',
        "inference.contig_as_guidepost=True",
        f'contigmap.contig_atoms={contig_atoms}',
        f"inference.num_designs={num_designs}",
        "inference.design_startnum=0",
        f"inference.output_prefix={output_prefix}",
        "hydra.job_logging.root.level=WARN",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="aa", overrides=overrides)
    return cfg


# --- Extract atoms from (indep, px0, seq_t) for E_cat scoring --------------

def indep_to_atom_dicts(indep, xyz_curr, seq_curr, cofactor_atoms_world):
    """Convert (xyz, seq) at a single time step into E_cat-compatible dicts.

    xyz_curr: (L, n_atoms, 3) tensor — usually px0 or x_t
    seq_curr: (L, NAATOKENS) one-hot or (L,) int tensor
    cofactor_atoms_world: list of HETATM dicts from the original motif.pdb,
                           untouched (cofactor is rigid).
    """
    if xyz_curr.dim() == 4:   # batch dim
        xyz_curr = xyz_curr[0]
    if seq_curr.dim() == 2:   # one-hot -> argmax
        seq_idx = seq_curr.argmax(-1)
    else:
        seq_idx = seq_curr
    seq_np = seq_idx.detach().cpu().numpy()
    xyz_np = xyz_curr.detach().cpu().numpy()
    is_sm = indep.is_sm.detach().cpu().numpy()

    out = []
    for i in range(xyz_np.shape[0]):
        if is_sm[i]: continue                # cofactor / ligand → use world version below
        aa = int(seq_np[i]) if seq_np[i] < 20 else 0
        rn = aa_idx_to_resname(aa)
        # Backbone heavies: N=0, CA=1, C=2, O=3 (RFD2 convention)
        for j, name in enumerate(["N", "CA", "C", "O"]):
            if j >= xyz_np.shape[1]: break
            p = xyz_np[i, j]
            if np.isnan(p).any(): continue
            out.append({"record": "ATOM", "name": name, "element": name[0],
                        "resname": rn, "chain": "A", "resseq": i + 1,
                        "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    # Append cofactor (HETATM) atoms unchanged — they live in WORLD coords already
    out.extend(cofactor_atoms_world)
    return out


def load_cofactor_world(motif_pdb: str):
    out = []
    for line in open(motif_pdb):
        if line[:6].strip() != "HETATM": continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            out.append({"record":"HETATM","name":name,"element":el,
                        "resname":line[17:20].strip(),"chain":line[21].strip(),
                        "resseq":int(line[22:26]),
                        "x":float(line[30:38]),"y":float(line[38:46]),"z":float(line[46:54])})
        except ValueError: continue
    return out


# --- The SMC loop ----------------------------------------------------------

def run_in_denoiser_smc(conf, fields, K=4, checkpoint_every=10,
                        lambda_max=1.0, schedule="linear",
                        ess_threshold_frac=0.5, tiered_lambdas=False,
                        seed=0, out_dir=".", verbose=True):
    """In-denoiser SMC. K independent particles share the same model+conf but
    have separate (indep, rfo, extra, features_cache) state. At each checkpoint
    we score px0 with E_cat and resample particles.
    """
    rng = random.Random(seed)
    cofactor_world = load_cofactor_world(conf.inference.input_pdb)

    # --- build K samplers and their initial states ------------------------
    # Note: model_runners.sampler_selector reads from conf; we re-seed inside
    # the loop so each particle gets independent noise even with the same conf.
    samplers = []
    states = []
    log_weights = [0.0] * K
    for k in range(K):
        if conf.inference.deterministic:
            torch.manual_seed(seed + k)
            np.random.seed(seed + k)
            random.seed(seed + k)
        # build a fresh sampler per particle (avoids stateful conflicts)
        sampler_k = model_runners.sampler_selector(conf)
        samplers.append(sampler_k)
        indep, contig_map, atomizer, t_step_input = sampler_k.sample_init(i_des=k)
        # init RFD2 features cache + extra — use the SAMPLER's _conf, not the
        # top-level conf, because the sampler composes extra Hydra defaults and
        # populates extra_tXd_names there. Using top-level conf yields an empty
        # list and KeyError: 'radius_of_gyration_v2' inside sample_step.
        # Matches run_inference.py:714 exactly.
        extra_tXd_names = getattr(sampler_k._conf, 'extra_tXd', [])
        features_cache = rfd_features.init_tXd_inference(
            indep, extra_tXd_names,
            sampler_k._conf.extra_tXd_params,
            sampler_k._conf.inference.conditions)
        states.append({
            "indep": indep, "contig_map": contig_map, "atomizer": atomizer,
            "t_step_input": t_step_input, "features_cache": features_cache,
            "rfo": None,
            "extra": {"rfo_uncond": None, "rfo_cond": None, "n_steps": None},
            # trajectory storage for save_outputs
            "px0_stack": [], "denoised_stack": [], "seq_stack": [],
            "traj_stack": defaultdict(list),
        })

    T = states[0]["t_step_input"]
    ts = torch.arange(int(T), samplers[0].inf_conf.final_step - 1, -1)
    n_steps = torch.ones(len(ts), dtype=int)
    history = []

    for it, t in enumerate(ts):
        if verbose:
            print(f"[step {it+1}/{len(ts)}] t={int(t)}", flush=True)
        # advance every particle
        for k in range(K):
            st = states[k]
            st["extra"]["n_steps"] = n_steps[it]
            px0, x_t, seq_t, rfo, extra = samplers[k].sample_step(
                int(t), st["indep"], st["rfo"], st["extra"], st["features_cache"])
            rf2aa.tensor_util.assert_same_shape(st["indep"].xyz, x_t)
            st["indep"].xyz = x_t
            st["rfo"] = rfo
            st["extra"] = extra
            st["px0_stack"].append(px0)
            st["denoised_stack"].append(copy.deepcopy(x_t))
            st["seq_stack"].append(seq_t)
            for kk, vv in extra.get("traj", {}).items():
                st["traj_stack"][kk].append(vv)

        # checkpoint: score + maybe resample
        is_checkpoint = (it % checkpoint_every == 0) or (t == ts[-1])
        if not is_checkpoint:
            continue

        # determine lambda(s)
        if tiered_lambdas:
            lam_dict = lambda_schedule_tiered(int(t), int(T))
            scalar_lam = None
        else:
            lam = (lambda_schedule_linear(int(t), int(T), lambda_max)
                   if schedule == "linear"
                   else lambda_schedule_sigmoid(int(t), int(T), lambda_max))
            lam_dict = None
            scalar_lam = lam

        # score each particle's current px0
        E_list = []
        for k in range(K):
            st = states[k]
            atoms = indep_to_atom_dicts(st["indep"], st["px0_stack"][-1],
                                          st["seq_stack"][-1], cofactor_world)
            if tiered_lambdas:
                E_k = e_cat_fn(atoms, fields, lambdas=lam_dict)
            else:
                E_k = e_cat_fn(atoms, fields)
            E_list.append(float(E_k))

        # weight + resample
        if tiered_lambdas:
            delta_logw = [-E for E in E_list]
        else:
            delta_logw = [-scalar_lam * E for E in E_list]
        log_weights = [lw + dlw for lw, dlw in zip(log_weights, delta_logw)]
        offset = max(log_weights)
        wnorm_raw = [math.exp(lw - offset) for lw in log_weights]
        s = sum(wnorm_raw)
        wnorm = [w / s for w in wnorm_raw]
        ess = 1.0 / sum(w * w for w in wnorm)
        do_resample = (lambda_max > 0 or tiered_lambdas) and ess < ess_threshold_frac * K

        entry = {
            "step": it, "t": int(t),
            "E_min": round(min(E_list), 4),
            "E_mean": round(sum(E_list) / K, 4),
            "E_max": round(max(E_list), 4),
            "ESS": round(ess, 3),
            "resampled": do_resample,
            "lambda": lam_dict if tiered_lambdas else scalar_lam,
        }
        history.append(entry)
        if verbose:
            print(f"   E[min,mean,max]=[{min(E_list):.3f}, {sum(E_list)/K:.3f}, {max(E_list):.3f}] "
                  f"ESS={ess:.2f}/{K}  {'RESAMPLED' if do_resample else 'kept'}", flush=True)

        if do_resample:
            parents = _resample_systematic(wnorm, rng)
            new_states = []
            for k in range(K):
                # deep-copy parent's state into a fresh slot for k
                p = parents[k]
                new_states.append(_deep_copy_state(states[p]))
            states = new_states
            log_weights = [0.0] * K

    # final outputs: call save_outputs per particle
    written = []
    from rf_diffusion.run_inference import save_outputs
    for k in range(K):
        st = states[k]
        sampler_k = samplers[k]
        prefix_k = os.path.join(out_dir, f"smc_p{k:03d}")
        # convert lists to stacks the way sample_one does
        denoised_xyz_stack = torch.flip(torch.stack(st["denoised_stack"]), [0])
        px0_xyz_stack     = torch.flip(torch.stack(st["px0_stack"]), [0])
        seq_stack         = list(reversed(st["seq_stack"]))
        traj_stack = {kk: torch.flip(torch.stack(vv), [0]) for kk, vv in st["traj_stack"].items()}
        ts_flipped = torch.flip(ts, [0])
        raw = (px0_xyz_stack, denoised_xyz_stack)

        # implicit sidechain + idealization + deatomize (lifted from sample_one)
        from rf_diffusion.run_inference import (
            add_implicit_side_chain_atoms, deatomize_sampler_outputs,
        )
        denoised_xyz_stack = add_implicit_side_chain_atoms(
            seq=st["indep"].seq, act_on_residue=~sampler_k.is_diffused,
            xyz=denoised_xyz_stack, xyz_with_sc=sampler_k.indep_orig.xyz)
        px0_filler = add_implicit_side_chain_atoms(
            seq=st["indep"].seq, act_on_residue=~sampler_k.is_diffused,
            xyz=px0_xyz_stack[..., :ChemData().NHEAVY, :],
            xyz_with_sc=sampler_k.indep_orig.xyz[..., :ChemData().NHEAVY, :])
        px0_xyz_stack[..., :ChemData().NHEAVY, :] = px0_filler
        is_protein = rf2aa.util.is_protein(st["indep"].seq)
        denoised_xyz_stack[:, is_protein] = idealize_backbone.idealize_bb_atoms(
            xyz=denoised_xyz_stack[:, is_protein], idx=st["indep"].idx[is_protein])
        px0_idealized = torch.clone(px0_xyz_stack)
        px0_idealized[:, is_protein] = idealize_backbone.idealize_bb_atoms(
            xyz=px0_xyz_stack[:, is_protein], idx=st["indep"].idx[is_protein])
        px0_xyz_stack = px0_idealized

        is_diffused = sampler_k.is_diffused.clone()
        atomizer = st["atomizer"]
        if atomizer is not None:
            indep_atomized = st["indep"].clone()
            is_diffused = atomize.deatomize_mask(atomizer, indep_atomized, is_diffused)
            indep_out, px0_xyz_stack, denoised_xyz_stack, seq_stack = \
                deatomize_sampler_outputs(atomizer, st["indep"],
                                            px0_xyz_stack, denoised_xyz_stack, seq_stack)
        else:
            indep_out = st["indep"]

        save_outputs(
            sampler_k, prefix_k, indep_out, st["contig_map"], atomizer,
            st["t_step_input"], denoised_xyz_stack, px0_xyz_stack, seq_stack,
            is_diffused, raw, traj_stack, ts_flipped,
        )
        written.append(prefix_k)

    return {"particles_written": written, "history": history,
            "final_log_weights": log_weights}


def _deep_copy_state(s):
    """Deep-copy a particle state dict. tensors are .clone()'d; defaultdicts copied."""
    new = {}
    new["indep"]          = s["indep"].clone() if hasattr(s["indep"], "clone") else copy.deepcopy(s["indep"])
    new["contig_map"]     = s["contig_map"]   # shared metadata — not modified during loop
    new["atomizer"]       = s["atomizer"]
    new["t_step_input"]   = s["t_step_input"]
    new["features_cache"] = s["features_cache"]   # cached features (immutable inputs)
    new["rfo"]            = copy.deepcopy(s["rfo"])
    new["extra"]          = copy.deepcopy(s["extra"])
    new["px0_stack"]      = [t.clone() for t in s["px0_stack"]]
    new["denoised_stack"] = [t.clone() for t in s["denoised_stack"]]
    new["seq_stack"]      = [t.clone() for t in s["seq_stack"]]
    new["traj_stack"]     = defaultdict(list)
    for k, v in s["traj_stack"].items():
        new["traj_stack"][k] = [t.clone() for t in v]
    return new


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", required=True)
    ap.add_argument("--a-cat", required=True)
    ap.add_argument("--ligand", default="LIG")
    ap.add_argument("--contigs", required=True)
    ap.add_argument("--contig-atoms", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("-K", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--schedule", choices=["linear","sigmoid"], default="linear")
    ap.add_argument("--ess-threshold-frac", type=float, default=0.5)
    ap.add_argument("--tiered-lambdas", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="seed RFD2 per-particle (vanilla preservation test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    output_prefix = os.path.join(args.out_dir, "smc")
    conf = build_conf(args.motif, args.ligand, args.contigs, args.contig_atoms,
                       args.ckpt, output_prefix, num_designs=args.K,
                       deterministic=args.deterministic)
    prepare_pyrosetta(conf)
    fields = load_fields(args.a_cat)

    print(f"# Stage 2 in-denoiser SMC")
    print(f"#   target={fields.target} mode={fields.mode}")
    print(f"#   K={args.K} checkpoint_every={args.checkpoint_every}")
    print(f"#   lambda_max={args.lambda_max} schedule={args.schedule} tiered={args.tiered_lambdas}")
    print(f"#   out_dir={args.out_dir}", flush=True)

    t0 = time.time()
    result = run_in_denoiser_smc(
        conf, fields, K=args.K, checkpoint_every=args.checkpoint_every,
        lambda_max=args.lambda_max, schedule=args.schedule,
        ess_threshold_frac=args.ess_threshold_frac,
        tiered_lambdas=args.tiered_lambdas, seed=args.seed,
        out_dir=args.out_dir, verbose=True,
    )
    print(f"# wallclock: {(time.time()-t0)/60:.1f} min")

    json.dump({"history": result["history"],
               "final_log_weights": result["final_log_weights"],
               "particles_written": result["particles_written"],
               "args": vars(args)},
              open(os.path.join(args.out_dir, "smc_log.json"), "w"), indent=2)
    print(f"# wrote {len(result['particles_written'])} particles + log to {args.out_dir}")


if __name__ == "__main__":
    main()
