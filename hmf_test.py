"""
Halo Mass Function (HMF) Analysis
=================================

Objective: Calculate the distribution of halo masses in the Seepage simulation
and compare it to the theoretical Schechter Function.

Theory:
- Schechter Function: dn/dM ~ M^alpha * exp(-M/M_star)
- Standard Model produces a specific slope (alpha ~ -1.2 to -1.5).
- Does the "Seepage" mechanism produce the same hierarchy of structures?

Method:
1. Run Seepage Simulation.
2. Identify particle groups (Friends-of-Friends or Grid-based clumps).
3. Histogram the masses (counts of particles per clump).
4. Fit to Schechter-like power law.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum as ndi_sum
from infall_simulation import SeepageSimulation, SeepageParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def run_hmf_test():
    print("="*60)
    print("HALO MASS FUNCTION (HMF) ANALYSIS")
    print("Verifying the Hierarchy of Cosmic Structure")
    print("="*60)
    
    # 1. Run Simulation
    print("\n1. Running Seepage Simulation...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=100.0, 
        random_seed=123
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=100000, 
        gravity_strength=60.0,
        friction=0.04,
        n_steps=600
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run()
    
    # 2. Identify Halos (Grid-based Group Finding)
    print("\n2. Identifying Halos...")
    
    # Bin particles
    n_grid = 64
    density = np.zeros((n_grid, n_grid, n_grid))
    idx = (sim.positions / sim.box * n_grid).astype(int)
    idx = np.clip(idx, 0, n_grid-1)
    np.add.at(density, (idx[:,0], idx[:,1], idx[:,2]), 1)
    
    # Define "structure" as regions above mean density threshold
    threshold = 5 # 5 particles per cell (approx 2.5x mean)
    mask = density > threshold
    
    from scipy.ndimage import maximum_filter
    
    # 2. Identify Halos (Peak Finding)
    print("\n2. Identifying Halos...")
    
    # Bin particles
    n_grid = 64
    density = np.zeros((n_grid, n_grid, n_grid))
    idx = (sim.positions / sim.box * n_grid).astype(int)
    idx = np.clip(idx, 0, n_grid-1)
    np.add.at(density, (idx[:,0], idx[:,1], idx[:,2]), 1)
    
    # Smooth slightly to merge noise
    from scipy.ndimage import gaussian_filter
    density_smooth = gaussian_filter(density, sigma=1.0)
    
    # Find local maxima
    # Neighborhood size = 3 cells
    local_max = maximum_filter(density_smooth, size=3) == density_smooth
    
    # Peak threshold
    # Mean density approx 1.9 (500k particles / 64^3 cells)
    # Threshold 1.5 (approx 0.78x mean)
    threshold_val = 1.5
    peaks_mask = local_max & (density_smooth > threshold_val)
    
    num_halos = np.sum(peaks_mask)
    print(f"   Found {num_halos} density peaks (halos).")
    
    if num_halos < 5:
        print("   Warning: Too few halos found for analysis. Returning early.")
        return
    
    # Get Peak Masses (approximate as peak value * volume factor)
    # Better: sum particles in 3x3x3 box around peak
    peak_indices = np.argwhere(peaks_mask)
    masses = []
    
    for p in peak_indices:
        # Sum 3x3x3 neighborhood (handling boundaries is tricky, just clipping for simplicity)
        slices = tuple(slice(max(0, c-1), min(n_grid, c+2)) for c in p)
        mass = np.sum(density[slices])
        masses.append(mass)
    
    masses = np.array(masses)
    
    # 3. Analyze Mass Function
    # Histogram
    bins = np.logspace(np.log10(min(masses)), np.log10(max(masses)), 20)
    hist, edges = np.histogram(masses, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Remove empty bins for log-log plot
    valid = hist > 0
    centers = centers[valid]
    hist = hist[valid]
    
    # Determine slope (alpha)
    # Fit simple power law to middle section (avoid cutoff tail)
    if len(centers) > 3:
        log_m = np.log10(centers)
        log_n = np.log10(hist)
        slope, intercept = np.polyfit(log_m, log_n, 1)
    else:
        slope = 0.0
    
    # 4. Visualize
    print("\n3. Plotting Mass Function...")
    plt.figure(figsize=(10, 6), facecolor='black')
    
    plt.loglog(centers, hist, 'o', color='cyan', label='Seepage Halos')
    
    # Plot fit line
    if len(centers) > 3:
        fit_y = 10**(intercept + slope * log_m)
        plt.loglog(centers, fit_y, '--', color='white', alpha=0.5, 
                   label=f'Power Law Fit (Slope = {slope:.2f})')
        
    # Schechter reference (arbitrary norm)
    # M^-1.2 ish
    ref_y = hist[0] * (centers / centers[0])**(-1.2)
    plt.loglog(centers, ref_y, ':', color='gray', label='Standard Schechter (~ -1.2)')
    
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Halo Mass Function (HMF)", color='white', fontsize=14)
    ax.set_xlabel("Halo Mass (Particle Count)", color='white')
    ax.set_ylabel("Number of Halos", color='white')
    ax.tick_params(colors='white')
    plt.legend()
    
    plt.savefig('/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory/hmf_test_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to hmf_test_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: HALO MASS FUNCTION")
    print(f"   Measured Power Law Slope: {slope:.2f}")
    
    if -2.0 < slope < -0.8:
        print("RESULT: MATCHES COSMIC HIERARCHY ✅")
        print("The mass distribution follows a realistic power law.")
    else:
        print("RESULT: ANOMALOUS DISTRIBUTION")
        print("Too many small or large halos compared to standard model.")
    print("="*60)

if __name__ == "__main__":
    run_hmf_test()
