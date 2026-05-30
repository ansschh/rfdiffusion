# F_cat — Catalytic-Fact Coverage Table

For each Tier-A target, every catalytic fact `f_i` is tagged with **which `A_cat` channel carries it** (or "—" if no channel does today), **which validator catches violations**, and a one-word **status**: ✓ (encoded), ◐ (approximated/uncertain), ✗ (missing).

The honest read: the project today encodes static cofactor sterics + active-species geometry + (newly) substrate-path geometry well; it does **not** carry electrostatic preorganization, dynamics, solvent, or stereochemical face selection. Static `A_cat` is the right starting object but is provably insufficient on its own — the table makes that visible per fact.

Channel legend: `A_steric` · `A_contact` · `A_path` · `A_anchor` · `A_TS` · `A_elec?` · `A_uncertainty`. Validators: `V_chem` (gated grammar on cofactor geometry) · `V_rxn` (reaction-geometry on the design) · `V_site` (metal-site fidelity, Boltz) · `V_preorg` (ensemble stability, future) · `V_fold` (Boltz CA-RMSD).

---

## 3ZP9 — Cp\*Ir(III) asymmetric transfer hydrogenation (imine → (S)-salsolidine)

| `f_i` | A_cat channel | Validator | Status |
|---|---|---|---|
| Metal identity & oxidation state Ir(III) | — (in `S_chem`) | `V_chem` G_metal | ✓ |
| η5 Cp\* hapticity (5 ring C at 2.0–2.2 Å) | `A_steric` | `V_chem` G_hapticity | ✓ |
| Bidentate κ-N,N chelate (picolinamide N3, N21) | — (in `S_chem`) | `V_chem` G_coord | ✓ |
| Hydride at open coordination (1.6 Å) | — (in `S_chem`, `active_species`) | `V_chem` G_active_state | ✓ |
| Substrate (imine C) approach vector from outside | `A_path` | `V_rxn` open-cone | ◐ (newly added as dummy + cone) |
| Hydride–imine alignment (TS geometry) | `A_TS` | `V_rxn` distance + angle | ◐ (transferred g‡, LOW confidence) |
| Stereochemical face (Si vs Re) | `A_TS` (orientation) | `V_rxn` face-vector | ✗ (compiler has no face spec) |
| Hydrophobic groove orientation (PHE/VAL/PRO/LEU) | `A_contact` + `A_anchor` | LigandMPNN-side, no validator today | ✗ |
| Anchor: aryl-sulfonamide-N → CA-II Zn (dative) | `A_anchor` (could carry) | — | ✗ (Rev1 diagnostic only) |
| Formate-assisted activation (β-H elimination) | — | — | ✗ (kinetics, not geometry) |
| Electrostatic preorganization (cationic imine TS) | `A_elec?` | — | ✗ |
| Active-species dynamics (turnover, NADH-like cycle) | — | `V_preorg` (future) | ✗ |
| Solvent / proton shuttle for formate | — | — | ✗ |

---

## 3WJC — CpRh(I) phenylacetylene polymerization

| `f_i` | A_cat channel | Validator | Status |
|---|---|---|---|
| Metal identity & oxidation state Rh(I) | — (in `S_chem`) | `V_chem` G_metal | ✓ |
| η5 Cp hapticity (5 B-ring C) | `A_steric` | `V_chem` G_hapticity | ✓ |
| COD diene = labile (drops on activation) | — (in `S_chem`, `L_labile`) | `V_chem` G_active_state | ✓ |
| Open face for monomer η2-coordination | `A_path` | `V_rxn` open-cone | ◐ |
| Alkynyl insertion geometry (M–Cα–Cβ angle) | `A_TS` | `V_rxn` insertion-vector | ✗ (compiler has no alkyne dummy) |
| Polymer chain growth direction | `A_path` (chain trajectory) | — | ✗ (dynamic, multi-step) |
| Stereoregularity (cis-syndiotactic vs cis-isotactic) | `A_TS` (face) | — | ✗ |
| Hydrophobic wall (LEU/VAL/ILE residues) | `A_contact` | LigandMPNN-side | ✗ |
| Anchor: covalent maleimide–Cys96 | `A_anchor` | — | ✗ (Rev1 diagnostic) |
| Polymer-host strain (polymer must not jam pocket) | — | `V_preorg` (future) | ✗ |
| Electrostatic stabilization of cationic Rh intermediate | `A_elec?` | — | ✗ |

---

## 5L8D — Ru(II) octahedral bpza oxychlorination

| `f_i` | A_cat channel | Validator | Status |
|---|---|---|---|
| Metal identity & oxidation state Ru(II) | — (in `S_chem`) | `V_chem` G_metal | ✓ |
| Octahedral geometry, **no Cp** | — (in `S_chem`) | `V_chem` G_coord + G_hapticity (forbidden) | ✓ |
| Three κ-N,N,O donors (bpza pyrazolyl + carboxylate) | — | `V_chem` G_coord | ✓ |
| CO retained, second CO drops on activation | — (in `S_chem`) | `V_chem` G_active_state | ✓ |
| Aqua at oxidant-binding open site (synthesized) | `A_path` (oxidant approach) | `V_rxn` open-cone | ◐ |
| Alkene η2-coordination geometry | `A_TS` | `V_rxn` alkene-vector | ✗ (compiler has no alkene dummy) |
| Cl⁻ trans positioning (oxychlorination requires) | `A_TS` | `V_rxn` Cl trans-angle | ✗ |
| Hydrophilic pocket (TRP × 2, TYR) for alkene | `A_contact` | LigandMPNN-side | ✗ |
| Anchor: bpza-carboxylate–Arg137 salt bridge | `A_anchor` + `A_elec?` | — | ✗ (electrostatic + diagnostic) |
| Solvent (water as oxygen source / hypochlorite delivery) | — | — | ✗ |
| Stereochemistry of vicinal chlorohydrin | `A_TS` | — | ✗ |

---

## 5OD5 — Cp\*Ir(III) transfer hydrogenation, His-coordinated open leg

| `f_i` | A_cat channel | Validator | Status |
|---|---|---|---|
| Metal identity & oxidation state Ir(III) | — (in `S_chem`) | `V_chem` G_metal | ✓ |
| η5 Cp\* hapticity | `A_steric` | `V_chem` G_hapticity | ✓ |
| κ-N,N chelate (azotochelin N donors) | — | `V_chem` G_coord | ✓ |
| **His227 coordinates open leg** (no synthesized hydride) | `A_anchor` (residue-specific) + `S_chem` (His-required donor) | `V_chem` G_active_state | ◐ (His as guidepost is privileged info) |
| Substrate (imine) approach displaces His? OR adjacent? | `A_path` + `A_TS` | `V_rxn` | ✗ (mechanism not pinned in curation) |
| Stereochemical face | `A_TS` | `V_rxn` | ✗ |
| Anchor: Fe-siderophore → CeuE (redox-switchable) | `A_anchor` + `A_elec?` | — | ✗ (dynamic anchor) |
| Cp\* disorder in crystal (occ 0.16–0.63) | `A_uncertainty` | — | ◐ (use best-resolved RIR copy) |
| Stereocontrol residue (H227A → ee 35% → 3%) | `A_contact` (load-bearing) | — | ✗ (sequence-level) |
| Activation state (precatalyst vs in-cycle) | — | — | ✗ |

---

## Cross-target summary — what static `A_cat` does and does not carry

**Carries today (✓):** metal identity, hapticity, coordination geometry, retained-donor set, active-species atom presence, substrate approach vector (newly), open-cone accessibility (newly).

**Approximated (◐):** TS geometry (LOW confidence, transferred analogs across all four), substrate atom (single dummy + cone — does not encode chemistry of forming bond), Cp\* steric body without η5 π-electronics.

**Missing (✗) — the project's known gap:** stereochemical face selection; LigandMPNN sequence-level chemistry (His donor here, avoid Cys near metal); electrostatic preorganization (cationic TS, salt bridge anchors); dynamics (turnover, conformational exchange, active-species generation); solvent (proton shuttles, oxygen source); polymer-host strain; ensemble stability.

**Implication for `H_A` (is `A_cat` sufficient?):** static `A_cat` carries the *necessary* sterics + path + active-species geometry — that is what retrieval can match. It does **not** carry the electrostatic / dynamic / face-selection facts that catalysis often hinges on. The brutal validation set therefore needs catalytic controls that probe **A_path** (rotate substrate vector), **G_hapticity** (wrong hapticity), and **active-species atom presence** (remove hydride) — which `V_chem` v0 + `V_rxn` v0 will catch. Controls that probe stereochemistry, electrostatics, and dynamics will **silently fail** to discriminate at v0 — and that is itself a finding worth surfacing rather than papering over.
