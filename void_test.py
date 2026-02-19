"""
Cosmic Void Density Analysis (The "Dirty Void" Test)
===================================================

Objective: Compare the emptiness of cosmic voids in the Dark Scaffold model vs
Standard Lambda-CDM predictions.

Prediction:
- Standard Model: Voids are extremely empty (matter evacuated by expansion).
- Dark Scaffold: Voids should be "dirtier" (contain more debris) because matter
  is seeping into filaments from a uniform background, and some might get left behind
  or take longer to fall in.

Method:
1. Run Seepage Simulation (to z=0).
2. Identify Voids (regions of lowest DM density).
3. Measure Baryon density within these void regions.
4. Compare with "Standard" expectation (approx < 10% mean density).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from infall_simulation import SeepageSimulation, SeepageParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def run_void_test():
    print("="*60)
    print("COSMIC VOID ANALYSIS")
    print("Checking for 'Dirty Voids' (Seepage Pattern Signature)")
    print("="*60)
    
    # 1. Run Seepage Simulation to Present Day
    print("\n1. Simulating Full Cosmic History...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=100.0, 
        random_seed=42
    ))
    scaffold.generate()
    dm_field = scaffold.density_field
    
    params = SeepageParams(
        n_particles=100000, 
        gravity_strength=50.0, # Standard coupling
        friction=0.04,
        n_steps=600 # Full run to z=0
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run() # Run full simulation
    
    # 2. Identify Voids in Dark Matter
    print("\n2. Identifying Deep Voids in Scaffold...")
    # Smooth to find large scale voids
    dm_smooth = gaussian_filter(dm_field, sigma=2.0)
    mean_dm = np.mean(dm_smooth)
    
    # Definition of Void: Regions with DM density < 20% of mean
    void_threshold = 0.2 * mean_dm
    void_mask = dm_smooth < void_threshold
    void_volume_fraction = np.sum(void_mask) / void_mask.size
    print(f"   Voids occupy {void_volume_fraction*100:.1f}% of volume.")
    
    # 3. Measure Baryon Density in Voids
    print("\n3. Measuring Baryon Content in Voids...")
    
    # Map particles to grid
    n_grid = 64
    baryon_density = np.zeros((n_grid, n_grid, n_grid))
    idx = (sim.positions / sim.box * n_grid).astype(int)
    idx = np.clip(idx, 0, n_grid-1)
    np.add.at(baryon_density, (idx[:,0], idx[:,1], idx[:,2]), 1)
    
    # Normalize baryon density relative to mean
    mean_baryon = np.mean(baryon_density)
    baryon_density_norm = baryon_density / (mean_baryon + 1e-10)
    
    # Get density distribution INSIDE voids
    void_baryon_densities = baryon_density_norm[void_mask]
    
    # Metrics
    mean_void_density = np.mean(void_baryon_densities)
    max_void_density = np.max(void_baryon_densities)
    empty_fraction = np.sum(void_baryon_densities == 0) / len(void_baryon_densities)
    
    print(f"   Mean Baryon Density in Voids: {mean_void_density:.3f} x mean")
    print(f"   Fraction of Void Cells Empty: {empty_fraction*100:.1f}%")
    
    # Standard Model Comparison (Approximation)
    # Standard voids are very empty, density typically < 0.1 mean
    standard_prediction = 0.1
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
    
    # Histogram of Void Densities
    plt.hist(void_baryon_densities, bins=50, range=(0, 2.0), 
             color='cyan', alpha=0.7, density=True, label='Dark Scaffold Voids')
    
    # Reference line for standard model expectation
    plt.axvline(standard_prediction, color='white', linestyle='--', linewidth=2, 
                label='Standard Model Limit (~0.1)')
    
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Density of Matter inside Cosmic Voids", color='white', fontsize=14)
    ax.set_xlabel("Density (Relative to Cosmic Mean)", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.tick_params(colors='white')
    plt.legend()
    
    plt.savefig('/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory/void_test_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to void_test_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: VOID STRUCTURE ANALYSIS")
    if mean_void_density > 0.15:
        print("RESULT: DIRTY VOIDS DETECTED ✅")
        print(f"Void density ({mean_void_density:.2f}) is higher than standard ({standard_prediction}).")
        print("This confirms the 'Seepage' prediction: matter is still falling in.")
    else:
        print("RESULT: CLEAN VOIDS")
        print("Seepage is extremely efficient, mimicking standard expansion clearing.")
    print("="*60)

if __name__ == "__main__":
    run_void_test()
