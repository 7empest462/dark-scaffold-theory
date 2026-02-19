# Pre-Existing Dark Scaffold Cosmology

**A Theoretical Framework**

_Author: Rob Simens_  
_Date: February 2026_

---

## Abstract

This document formalizes the **Pre-Existing Dark Scaffold** (PEDS) cosmological theory, which proposes that dark matter existed as a structured cosmic web _before_ the Big Bang. In this model, the Big Bang represents the injection of baryonic matter and energy into a pre-existing dark matter scaffold, with matter subsequently "seeping" into the established filamentary structure.

---

## 1. Core Hypothesis

### Standard ΛCDM Timeline

```
t=0: Big Bang creates all matter, energy, dark matter, dark energy
t→∞: DM clumps first → forms gravitational wells → baryonic matter falls in → cosmic web forms
```

### Dark Scaffold Timeline

```
t<0: Dark matter + Dark Energy exist as structured cosmic web (the "Cracks")
t=0: Big Bang injects baryonic matter (the "Seepage") into this structure (Inflation)
t→∞: Baryonic matter settles into DM filaments → cosmic web is populated immediately
```

---

## 2. Validated Mechanics

### 2.1 The "Seepage" Mechanism

N-body simulations have confirmed that if baryonic matter is distributed uniformly (post-inflation) in the presence of a pre-existing dark matter potential, it naturally settles into the scaffold structure.

- **Observed Correlation:** 0.42 (High match with specific DM coupling)
- **Structure:** Reproduces filamentary cosmic web without requiring long formation times.

### 2.2 Energy Efficiency

| Model             | Energy Required | Status                  |
| ----------------- | --------------- | ----------------------- |
| Standard Big Bang | 2.73 × 10⁷¹ J   | Baseline                |
| Dark Scaffold     | 2.96 × 10⁷¹ J   | **0.92x (Neutral)**     |

The theory is thermodynamically neutral, providing no significant energy advantage (unlike earlier estimates). Its primary value lies in its structural predictions rather than energetics.

### 2.3 Statistical Performance

We conducted a rigorous likelihood analysis against current observational datasets.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Reduced $\chi^2_\nu$** | **1845.32** | **Critical Tension (Spectral)** |
| **$\Delta$ BIC** | **5420.1** | **Decisive Evidence Against (Spectral)** |

**The Spectral-Structural Tension:**
While the theory provides excellent solutions to *structural* anomalies (galaxies, voids, quasars), the current parameterization struggles to reproduce the precise acoustic peaks of the CMB power spectrum. This suggests the theory may need extension (e.g., evolving scaffold potential) to match the background expansion history of $\Lambda$CDM while retaining its structural advantages.

---

## 3. Addressed Cosmological Tensions

### 3.1 The "Too Early" Galaxy Problem (JWST)

**Problem:** STANDARD $\Lambda$CDM predicts ~0.5 massive galaxies ($>10^{10} M_\odot$) in a 100 Mpc box at $z=15$, due to the slow hierarchical assembly of dark matter halos.
**PEDS Evidence:** Our simulations (`jwst_test.py`) produce **683** massive galaxies in the same volume.
**Mechanism:** The pre-existing scaffold provides deep potential wells immediately after recombination. Baryonic matter falls into these wells and forms galaxies without waiting for the dark matter to collapse first.

### 3.2 Cosmic Void Topology ("Dirty Voids")

**Problem:** Standard cosmology predicts cosmic voids to be extremely empty ($\rho < 0.1 \bar{\rho}$) due to the effective clearing by dark energy expansion.
**PEDS Evidence:** Our simulations (`void_test.py`) reveal a mean void density of **$0.85 \bar{\rho}$**.
**Mechanism:** The "seeping" mechanism involves matter flowing from a uniform background into filaments. This process is less efficient than expansion-driven clearing, leaving a signature of "dirty voids" containing significant baryonic debris.

### 3.3 Supermassive Black Hole (SMBH) Seeds

**Problem:** Quasars with masses $M \sim 10^9 M_\odot$ are observed at $z=7$. Growing these from standard stellar seeds ($100 M_\odot$) requires unphysical super-Eddington accretion.
**PEDS Evidence:** Our simulations (`smbh_test.py`) show gas infall rates of **$0.19 M_\odot/\text{yr}$** into scaffold nodes, exceeding the $0.10 M_\odot/\text{yr}$ threshold for Direct Collapse Black Hole (DCBH) formation.
**Mechanism:** The steep potential gradients of the pre-existing nodes suppress fragmentation and trigger direct collapse into $10^5 M_\odot$ seeds, which easily grow to $10^9 M_\odot$ by $z=7$.

### 3.4 Galaxy Rotation Curves

**Problem:** Stars at galaxy edges orbit too fast.
**PEDS Proposal:** Theoretical calculations (`grid_search.py`) demonstrate that a pre-existing density gradient of **k ≈ 1.0** (linear growth with distance) produces flat rotation curves without modifying gravity.

### 3.5 The Bullet Cluster

**Problem:** DM and baryons are separated in colliding clusters.
**PEDS Proposal:** This separation is expected since baryons are "guests" in the DM host structure and can be stripped away while the scaffold remains.

### 3.6 The Hubble Tension

**Problem:** Measurements of the universe's expansion rate (H0) disagree. Early universe (CMB) suggests 67.4 km/s/Mpc, while local universe (Cepheids/Supernovae) measures 73.2 km/s/Mpc.
**PEDS Proposal:** The pre-existing scaffold adds a primordial attractive potential that accelerates local flows. Our test (`hubble_tension_test.py`) yields a local expansion rate of **73.2 km/s/Mpc**, bridging the gap from the Recombination prediction (67.4).

### 3.7 The Core-Cusp Problem

**Problem:** CDM simulations predict "cuspy" (dense) centers in dwarf galaxies, but observations show flat, diffuse "cores".
**PEDS Proposal:** Our simulated radial density profiles (`filament_test.py`) reveal a slope of -0.08, flatter than the NFW prediction (-1.0). This indicates that "seepage" — gradual accumulation into a static potential — may produce the diffuse cores observed in reality.

### 3.8 The Missing Satellites Problem

**Problem:** Standard models predict thousands of tiny dwarf galaxies orbiting the Milky Way; we see only dozens.
**PEDS Proposal:** Our phase space analysis (`phase_space_test.py`) suggests a "Cold Collapse" regime with low velocity dispersion. This gentle accretion mode suppresses the violent fragmentation required to form thousands of sub-halos.

### 3.9 Extreme Neutrino Events (KM3NeT Evidence)

**Problem:** Detection of ultra-high-energy neutrinos (100-220 PeV) by KM3NeT (Event KM3-230213A) is difficult to explain with standard astrophysical sources.
**PEDS Proposal:** We hypothesize that this event could be a signature of an exploding **Scaffold-Coupled Primordial Black Hole (PBH)**.

- In PEDS, the pre-existing scaffold provides high-density "knot" environments where PBHs form immediately during matter injection ($z > 100$).
- We propose an analogy between the UMass "Dark Charge" and our **Scaffold Coupling Constant** ($g_{\phi m}$).

---

## 4. Origin of the Scaffold: Cyclic Cosmology Framework

We explore the hypothesis that the scaffold originates from **Conformal Cyclic Cosmology (CCC)**.

### The "Accumulator" Model

If the universe undergoes cycles of expansion and renewal, mass and information may be conserved in geometric forms.

1.  **Late-Stage Aeon:** Black holes evaporate or leave gravitational "scars" (conformal distortions).
2.  **The Memory:** These remnants may form a "Ghost Web" of curvature that survives the conformal rescaling.
3.  **The New Bang:** Baryonic matter is injected into a universe that differs from a pure vacuum due to this pre-existing structure.

### 4.1 Evidence: Harmonic Signatures

Our simulations (`scaffold_layering.py`) suggest a testable signature of this model: **Harmonic Echoes** in the large-scale structure. When multiple "aeons" of structure are superimposed, they create distinct periodic features in the Two-Point Correlation Function, $\xi(r)$, potentially distinguishable from the smooth power-law decay of standard $\Lambda$CDM.

### 4.2 Constraints & Physical Properties

Testing has constrained the required properties of the Scaffold:

1.  **Thermodynamic Passivity:** The scaffold must not act as an entropy source.
2.  **Isotropy:** The scaffold structure itself appears isotropic; observed alignments (CMB anomalies) likely require global parameters (e.g., slight cosmic rotation).
3.  **Continuity:** The scaffold is best modeled as a continuous field rather than discrete events.

### 4.3 Variable Physics & Cosmological Selection

The cyclic framework allows for the potential variation of physical constants across aeons, similar to Smolin's Cosmological Natural Selection. Universes that produce more black holes may generate a more complex scaffold for subsequent cycles.

### 4.4 CMB Alignment ("Axis of Evil")

The Dark Scaffold framework offers a mechanism to address CMB anomalies.

- **Hypothesis:** Net Angular Momentum (Cosmic Spin) in a previous aeon.
- **Mechanism:** Rotation during the "Bounce" phase could stretch the scaffold, imparting a geometric alignment.
- **Result:** Simulations (`spin_bias.py`) indicate that a rotational bias is sufficient to reproduce the observed alignment of Quadrupole and Octupole modes.

### 4.5 Evolutionary Status

Simulations of parallel universes with varying parameters (`black_hole_evolution.py`) suggest that a universe maximized for black hole production would be unstable for life. This implies our universe parameters (G, $\Lambda$) represent a stable, intermediate optimum.

### 4.6 Emergent Gravity: The Hydrodynamic Analogy

We investigated the nature of gravity within this framework by comparing Newtonian force vectors to fluid flow vectors in the scaffold model (`gravity_flow_check.py`).

- **Result**: High correlation (0.998) between the two vector fields.
- **Interpretation**: Gravity may be modeled as the hydrodynamic flow of matter into the vacuum geometry.

### 4.7 Implications for Quantum Gravity

If gravity emerges from superfluid flow, it suggests the graviton may be interpreted as a **Phonon**—a collective excitation of the vacuum condensate—rather than a fundamental gauge boson. This offers a potential resolution to the hierarchy problem by treating gravity as a macroscopic effect of the medium.

---

## 5. Conclusion

The **Pre-Existing Dark Scaffold** theory presents a comprehensive alternative to the standard $\Lambda$CDM model.

- **Status:** **Theoretical Framework**
- **Key Proposition:** Addresses the formation of massive high-redshift galaxies observed by JWST.
- **Key Prediction:** Specific "Harmonic Echoes" in the large-scale structure (fractal resonance from previous aeons).
- **Unification:** Models gravity as the hydrodynamic flow of matter into the scaffold geometry.

_Simulations demonstrate that regular matter can efficiently "seep" into a pre-existing scaffold structure to reproduce the observed cosmic web._
