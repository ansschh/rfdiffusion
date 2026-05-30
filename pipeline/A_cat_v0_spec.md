# A_cat v0 — typed-Gaussian / vectors+cones representation

Worked schema for 3ZP9 as the canonical example. No code yet; this is the contract that
the compiler's `C_η(X_chem) → A_cat` step will emit, that retrieval will score against,
and that future RFD2 guidance can condition on. Designed so a single A_cat object serves
**retrieval, RFD constraints, and validation** without re-derivation (PI 2026-05-29).

---

## 1. Local frame

A_cat is expressed in a **reactive-core local frame** so that retrieval against an arbitrary
candidate pocket is a rigid-body alignment problem.

- **Origin:** metal atom position (for single-metal cofactors). For multi-metal: centroid of
  catalytic metals, weighted by oxidation-state.
- **+z axis (`v_open`):** outward from the metal along the open-coordination vector. For a
  hydride-bearing piano-stool: `v_open = normalize(metal → hydride)`. For an aqua-bearing
  octahedral: `v_open = normalize(metal → aqua-O)`. For His-coordinated: `v_open = normalize(metal → His-N)`
  with sign flipped (substrate would displace His).
- **+x axis:** projection of (metal → centroid of retained κ-N,N donors) onto the plane
  perpendicular to z.
- **+y axis:** `z × x` (right-handed).
- **Symmetric complexes (degenerate frames):** store the **set of all symmetry-equivalent
  frames** as alternative orientations of A_cat; retrieval scores against the best-matching
  one (R, t over the union).

For 3ZP9: origin = IR; +z = IR→H1 (synthesized hydride, 1.6 Å); +x = projection of IR→midpoint(N3, N21).

## 2. Channels

A_cat is a bundle (not one field). Each channel either carries information today (✓), partially
(◐), or is marked missing (✗). The `F_cat_coverage.md` per-target audit tracks status.

Each channel is a finite set of components — typed Gaussians for diffuse densities, point + radius
for spheres of avoidance, vector + half-angle for directional cones. **No voxel grids.**

| Channel | Geometry | What it carries | Source rule (from `S_chem`) | Validator |
|---|---|---|---|---|
| `A_steric` | Spheres at cofactor heavy-atom positions, radii from vdW + tolerance | Cofactor exclusion volume (where protein cannot intrude) | Direct from compiled motif geometry | `V_chem` G_coord (clash) |
| `A_contact` | Typed Gaussians (centers near cofactor surface, soft Σ) | Where protein side chains should pack; per-type (hydrophobic / polar / aromatic) | Heuristic from cofactor surface chemistry + reaction class | (LigandMPNN-side, no validator v0) |
| `A_path` | Cone (apex=metal, axis=+z, half-angle, extent) | Substrate approach region — must remain accessible | Reaction-class spec (ATH: cone along +z; oxychlor: cone along oxidant-binding vector) | `V_rxn` G_access |
| `A_anchor` | Sphere(s) at intended anchoring residue position, radius | Where covalent / dative / supramolecular handle attaches | `S_chem.H_anchor` (Rev1 — diagnostic in v0) | (no validator v0) |
| `A_TS` | Point(s) at expected TS-atom positions + alignment vector (M → reactive-atom → cone) | TS geometry: M-reactive-atom distance, approach angle, stereochemical face | Transferred `g‡` analog (LOW confidence per curation) | `V_rxn` G_access (cone), `V_rxn` G_distance (when implemented) |
| `A_elec?` | (deferred) — would be charges + dipoles + H-bond donor/acceptor markers | Electrostatic preorganization | `S_chem.M.charge` + cofactor partial charges + TS charge | (no validator v0) |
| `A_uncertainty` | Per-component σ_pos and confidence label | Where the chemistry is solid vs transferred / disordered | curation `confidence` field per fact | meta — not a gate |

## 3. Concrete instantiation for 3ZP9 (Cp\*Ir(III) ATH)

```yaml
A_cat:
  target: 3ZP9
  frame:
    origin: IR
    v_open: "IR -> H1"          # synthesized hydride, 1.6 A
    v_x: "IR -> centroid(N3, N21)"
  channels:

    A_steric:                    # exclusion spheres at cofactor heavy atoms
      - {pos: IR,  r: 2.2}       # metal vdW + tolerance
      - {pos: N3,  r: 1.8}
      - {pos: N21, r: 1.8}
      - {pos: C1,  r: 1.8}       # Cp* ring carbons (5)
      - {pos: C3,  r: 1.8}
      - {pos: C5,  r: 1.8}
      - {pos: C7,  r: 1.8}
      - {pos: C9,  r: 1.8}
      - {pos: H1,  r: 1.4}

    A_contact:                   # typed Gaussians for residue packing preferences
      - {type: hydrophobic, mu_local: [3.5, 0.0, 2.5], Sigma_diag: [1.5, 1.5, 1.5], w: 1.0}   # Cp* face
      - {type: hydrophobic, mu_local: [0.0, 3.5, 2.5], Sigma_diag: [1.5, 1.5, 1.5], w: 1.0}
      - {type: polar,       mu_local: [3.0, -3.0, -1.0], Sigma_diag: [1.2, 1.2, 1.2], w: 0.5}   # sulfonamide-side

    A_path:                      # substrate (imine) approach cone — V_rxn enforces
      apex: H1
      axis_local: [0, 0, 1]      # +z (outward along v_open)
      half_angle_deg: 30
      extent_A: 5.5
      content_type: imine_C
      expected_M_substrate_A: [2.5, 3.5]   # imine C should approach to this distance

    A_anchor:                    # CA-II Zn anchor — diagnostic only (Rev1)
      - {residue_role: aryl_sulfonamide_N_to_Zn, sphere_center_local: [6.0, -4.0, -2.0], r: 2.5,
         carried: false, status: "diagnostic — de novo scaffold need not reuse"}

    A_TS:                        # transferred g-dd (Noyori-type 6-mem cyclic TS) — LOW confidence
      - {component: imine_C_position, pos_local: [0.0, 0.0, 3.0], sigma_A: 0.7, confidence: low}
      - {component: amine_N_position, pos_local: [-1.5, 0.0, 2.5], sigma_A: 0.8, confidence: low}
      - {component: hydride_transfer_vector, from: H1, to: imine_C, max_dist_A: 2.5}
      - {component: stereochemical_face, normal_local: [1.0, 0.0, 0.0], status: missing}   # ✗ no spec

    A_elec:
      status: missing            # ✗ no electrostatic channels in v0
      would_carry: ["cationic_imine_TS", "Ir(III)_partial_charge", "polar_groove_dipoles"]

    A_uncertainty:
      A_TS: low_confidence       # transferred g-dd
      A_anchor: diagnostic       # not load-bearing
      A_elec: missing            # ✗
      A_dynamics: missing        # ✗
      A_solvent: missing         # ✗
```

## 4. Retrieval scoring contract

Given query A_cat (above) and a candidate pocket B in some scaffold S, the rigid-body alignment
`(R, t)` is the variable; retrieval score is:

```
S(R, t; A_cat, B) =
   [hard gates: G_coord · G_hapticity · G_open-site · G_TS-access]                      (categorical)
 + w1 · Σ_τ ∫ A_contact_τ(x) B_τ(R x + t) dx                                            (soft contact overlap)
 + w2 · Σ ⟨A_anchor_i, nearest residue of allowed_type in B⟩                            (soft anchor)
 + w3 · S_electrostatic-ish(A_elec, B)                                                  (placeholder; 0 in v0)
 - w4 · E_clash(A_steric, B)                                                            (vdW overlap penalty)
 - w5 · E_path-block(A_path, B)                                                         (cone occupancy penalty)
```

Gates are computed first; if any gate fails the candidate is rejected regardless of soft score.
This is the same gate-then-soft structure as `V_chem` / `V_rxn`.

## 5. Per-channel derivation from `S_chem`

The compiler step `C_η(X_chem) → A_cat` derives each channel deterministically:

- `A_steric` ← cofactor heavy-atom coordinates + element vdW radii (lookup table) + reaction-class tolerance.
- `A_contact` ← reaction-class template (e.g. ATH groove = hydrophobic patch on Cp\* face) parameterized by cofactor geometry.
- `A_path` ← `S_chem.open_site_vector` + reaction-class cone parameters (half-angle, extent) from a small lookup.
- `A_anchor` ← `S_chem.H_anchor`; emitted at `carried: false` in v0 (diagnostic only per Rev1).
- `A_TS` ← `S_chem.g_TS_atoms` (the transferred g‡ analog atom positions in the reactive-core frame) + confidence tag.
- `A_elec` ← `S_chem.M.charge` + cofactor partial charges + TS charge. **Not built in v0** — channel slot present with `status: missing` so consumers know it's absent.
- `A_uncertainty` ← per-channel confidence tags from curation; consumers must propagate.

## 6. Honest limits (must be visible to consumers)

- **Static representation.** No ensemble / dynamics. A_cat says "leave this cone open at this
  geometry"; it does not say "stays open across motions."
- **No electrostatic preorganization in v0.** The retrieval score cannot distinguish a pocket
  that stabilizes a cationic TS from one that destabilizes it.
- **`A_TS` confidence is LOW for all four Tier-A targets** (transferred analogs, no DFT). The
  `A_uncertainty` channel surfaces this; consumers must weight accordingly.
- **No stereochemical face selection.** The compiler has no representation of R/S preference
  today; `A_TS.stereochemical_face` is `status: missing`.
- **No sequence-level chemistry.** `A_cat` enters retrieval and RFD constraints; it does NOT
  currently enter LigandMPNN. Sequence-level chemistry violations ("His donor here," "avoid
  Cys near metal") are deferred to post-design `V_chem` or future LigandMPNN logit-steering.

These limits are the explicit statement of `H_A`'s scope: A_cat carries sterics, contact
preferences, substrate path, partial TS geometry — and is **provably insufficient** alone for
catalysis. The brutal validation set (real vs catalytically-damaged A_cat under retrieval +
V_chem + V_rxn) is what tells us whether v0 is enough for the *structural* discrimination job
that retrieval is meant to do.
