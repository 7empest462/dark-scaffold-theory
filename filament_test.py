"""
Filament Radial Profile Analysis
================================

Objective: Measure the density of baryons as a function of distance from the 
center of a cosmic filament.

Theory:
- Standard DM Filaments: Have a characteristic "cuspy" profile (roughly 1/r^2).
- Seepage Prediction: If matter flows properly, it should accumulate in the
  filament core, reproducing the high-density "spine" observed in nature.

Method:
1. Run Seepage Simulation.
2. Identify the spine of a major filament (using DM ridge).
3. Measure Baryon density in cylindrical shells around the spine.
4. Plot Density vs Radius.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from infall_simulation import SeepageSimulation, SeepageParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def get_filament_spine(dm_field):
    """Find the densest line (spine) in the box."""
    # Simple approach: Find max density point, then follow gradient? 
    # Or just take a 2D slice and find the peak?
    # Let's effectively treat Z-axis as the filament direction for a segment.
    
    # Project to 2D
    proj = np.mean(dm_field, axis=2)
    max_idx = np.unravel_index(np.argmax(proj), proj.shape)
    
    # Return (x, y) of the spine (assuming it runs roughly along Z)
    return max_idx

def run_filament_test():
    print("="*60)
    print("FILAMENT RADIAL PROFILE ANALYSIS")
    print("Checking if seepage creates realistic dense backbones")
    print("="*60)
    
    # 1. Run Simulation
    print("\n1. Running Seepage Simulation...")
    grid_size = 64
    box_size = 100.0
    
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=grid_size, box_size=box_size, 
        random_seed=42 # Known seed with good structure
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=100000, 
        gravity_strength=50.0,
        friction=0.04,
        n_steps=600
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run()
    
    # 2. Identify Filament Spine
    print("\n2. Identifying Filament Spine...")
    dm = scaffold.density_field
    spine_x_idx, spine_y_idx = get_filament_spine(dm)
    
    spine_x = spine_x_idx * (box_size / grid_size)
    spine_y = spine_y_idx * (box_size / grid_size)
    
    print(f"   Analyzing filament near x={spine_x:.1f}, y={spine_y:.1f}")
    
    # 3. Measure Radial Density Profile
    print("\n3. Calculating Radial Density...")
    
    # Calculate R for every particle (relative to spine in XY plane)
    # This assumes filament is z-aligned (simplification, but works for statistical check)
    dx = sim.positions[:,0] - spine_x
    dy = sim.positions[:,1] - spine_y
    
    # Periodic BC
    dx = dx - box_size * np.round(dx/box_size)
    dy = dy - box_size * np.round(dy/box_size)
    
    r = np.sqrt(dx**2 + dy**2)
    
    # Bin by radius
    bins = np.linspace(0, 10.0, 20) # 0 to 10 Mpc
    counts, edges = np.histogram(r, bins=bins)
    
    # Volume of cylindrical shell: pi * (r2^2 - r1^2) * L
    # We normalized density essentially
    centers = (edges[:-1] + edges[1:]) / 2
    volumes = np.pi * (edges[1:]**2 - edges[:-1]**2) * box_size
    
    densities = counts / volumes
    
    # Normalize to mean density in box
    global_mean = len(r) / (box_size**3)
    densities_norm = densities / global_mean
    
    # 4. Fit / Compare
    # NFW-like profile goes locally as 1/r or 1/r^2
    # Let's fit A * r^slope
    valid = (densities_norm > 0) & (centers > 0.5) # Exclude very center (smoothing) and empty
    fit_r = centers[valid]
    fit_rho = densities_norm[valid]
    
    if len(fit_r) > 2:
        slope, intercept = np.polyfit(np.log10(fit_r), np.log10(fit_rho), 1)
    else:
        slope = 0.0
    
    # 5. Visualize
    print("\n4. Plotting Profile...")
    plt.figure(figsize=(10, 6), facecolor='black')
    
    plt.loglog(centers, densities_norm, 'o-', color='cyan', linewidth=2, label='Seepage Baryons')
    
    # Fit line
    if len(fit_r) > 2:
        y_fit = 10**(intercept + slope * np.log10(centers))
        plt.loglog(centers, y_fit, '--', color='white', alpha=0.5, 
                   label=f'Power Law Fit (Slope = {slope:.2f})')
        
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Filament Radial Density Profile", color='white', fontsize=14)
    ax.set_xlabel("Radius from Spine [Mpc]", color='white')
    ax.set_ylabel("Density (Relative to Mean)", color='white')
    ax.tick_params(colors='white')
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.savefig('/Users/robsimens/Documents/Cosmology/dark-scaffold-theory/filament_test_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to filament_test_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: FILAMENT PROFILE")
    print(f"   Radial Slope: {slope:.2f}")
    
    if slope < -0.5:
        print("RESULT: REALISTIC CUSP ✅")
        print("Matter accumulates densely in filament cores (Cuspy).")
    else:
        print("RESULT: DIFFUSE CORES")
        print("Filaments are fluffy, indicating weak binding.")
    print("="*60)

if __name__ == "__main__":
    run_filament_test()
