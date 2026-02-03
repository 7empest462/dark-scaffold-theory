# Pre-Existing Dark Scaffold Theory

## Progress Summary — February 2026

**Author:** Rob Simens  
**Status:** Active Research 🔬

---

## Core Hypothesis

The Big Bang didn't create dark matter — it injected baryonic matter into a **pre-existing dark matter scaffold**. Dark energy was also already present as a property of the vacuum.

```
BEFORE Big Bang: DM scaffold + Dark Energy exist
AT Big Bang:     Baryonic matter + radiation injected
AFTER:           Matter "seeps" into DM filaments
```

---

## What We Built

| Script                  | Purpose                       |
| ----------------------- | ----------------------------- |
| `scaffold_generator.py` | Gaussian Random Field DM web  |
| `seeping_simulation.py` | Basic particle seeping model  |
| `nbody_simulation.py`   | Full N-body with self-gravity |
| `energy_budget.py`      | Thermodynamic analysis        |
| `theory_likelihood.py`  | Observational comparison      |
| `run.sh`                | Runner script                 |

---

## Key Results

### ✅ Successes

| Observation           | Our Prediction | Observed Value | Status       |
| --------------------- | -------------- | -------------- | ------------ |
| Bullet Cluster offset | 149 kpc        | 150 ± 50 kpc   | ✅ Perfect   |
| BAO scale             | 145 Mpc        | 147 ± 2 Mpc    | ✅ Good      |
| Baryon fraction       | 15.8%          | 15.6 ± 1%      | ✅ Excellent |
| Energy requirement    | **20× less**   | —              | ✅ Major win |
| Flat rotation curves  | k ≈ 1.0 works  | —              | ✅ Solved    |

### ⚠️ In Progress

- DM-baryon correlation: 8% (needs improvement)
- Small-scale clustering metrics
- CMB detailed predictions

---

## Physical Insights

1. **Dark matter density grows linearly with distance** (your k-gradient) → flat rotation curves
2. **Gravitational torques** from asymmetric scaffold → galactic spin-up
3. **Pre-existing wells** → explains early massive galaxies (JWST)
4. **DM/baryon decoupling** → Bullet Cluster naturally explained

---

## Next Steps

- [ ] Tune N-body for higher DM-baryon correlation
- [ ] Add angular momentum tracking for spin-up
- [ ] CMB anisotropy predictions
- [ ] Compare with JWST early galaxy data
- [ ] Investigate scaffold origin mechanisms

---

## Run Commands

```bash
cd ~/Documents/Cosmology/dark-scaffold-theory
./run.sh scaffold    # Generate DM web
./run.sh nbody       # Run N-body simulation
./run.sh likelihood  # Run observational comparison
```

---

_This is a working document. Research continues._
