"""
Galactic Spin-Up Test
=====================

Objective: Verify if baryonic matter "seeping" into asymmetric DM scaffold nodes
naturally acquires angular momentum (spin).

Hypothesis: The filamentary structure provides anisotropic gravitational torques,
causing infalling matter to spin up without needing initial rotation.

Method:
1. Run SeepageSimulation (distributed start).
2. Identify DM density peaks (halo centers).
3. For each halo, calculate net angular momentum of trapped baryons.
4. Visualize spin vectors relative to filaments.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from infall_simulation import SeepageSimulation, SeepageParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def find_dm_halos(dm_field, box_size, threshold_std=2.0, min_dist_cells=5):
    """Identify density peaks in the DM scaffold."""
    # Thresholding
    threshold = np.mean(dm_field) + threshold_std * np.std(dm_field)
    mask = dm_field > threshold
    
    # Local maxima
    local_max = maximum_filter(dm_field, size=min_dist_cells) == dm_field
    peaks_mask = local_max & mask
    
    peak_indices = np.argwhere(peaks_mask)
    
    # Convert to physical coordinates
    # grid_size = dm_field.shape[0]
    # coords = peak_indices * (box_size / grid_size)
    
    return peak_indices

def calculate_halo_spin(halo_idx, positions, velocities, box_size, grid_size, radius_mpc=5.0):
    """Calculate angular momentum of particles within radius of halo center."""
    # Halo center in physical coords
    center = halo_idx * (box_size / grid_size)
    
    # Distances (periodic)
    delta = positions - center
    delta = delta - box_size * np.round(delta / box_size) # Periodic BC
    
    dist = np.linalg.norm(delta, axis=1)
    
    # Select particles in halo
    mask = dist < radius_mpc
    n_particles = np.sum(mask)
    
    if n_particles < 10:
        return None, 0
        
    p_halo = delta[mask]
    v_halo = velocities[mask]
    
    # Center of mass frame
    v_cm = np.mean(v_halo, axis=0)
    v_rel = v_halo - v_cm
    
    # Angular momentum: L = sum(r x v)
    # Mass assumed 1 for all particles
    L = np.cross(p_halo, v_rel)
    L_total = np.sum(L, axis=0)
    
    # Specific angular momentum (magnitude)
    j_mag = np.linalg.norm(L_total) / n_particles
    
    return L_total, j_mag

def run_spin_test():
    print("="*60)
    print("GALACTIC SPIN-UP TEST")
    print("Checking if scaffold induces rotation in seeping matter")
    print("="*60)
    
    # 1. Setup & Run Simulation
    print("\n1. Running Seepage Simulation...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=100.0, spectral_index=-1.5,
        smoothing_scale=2.0, filament_threshold=0.5, random_seed=123 
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=15000, # More particles for better statistics
        gravity_strength=50.0,
        friction=0.04,
        n_steps=800
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run()
    
    # 2. Analyze Spin
    print("\n2. Identifying Halos and Calculating Spin...")
    dm = scaffold.density_field
    halos = find_dm_halos(dm, sim.box)
    print(f"   Found {len(halos)} potential halos.")
    
    spins = []
    magnitudes = []
    valid_halos = []
    
    for h_idx in halos:
        L, mag = calculate_halo_spin(h_idx, sim.positions, sim.velocities, 
                                   sim.box, scaffold.params.grid_size)
        if L is not None:
            spins.append(L)
            magnitudes.append(mag)
            valid_halos.append(h_idx)
            
    avg_spin = np.mean(magnitudes)
    print(f"   Analyzed {len(valid_halos)} massive halos.")
    print(f"   Average Specific Angular Momentum: {avg_spin:.4f} Mpc·(Mpc/unit_time)")
    
    # 3. Visualize
    print("\n3. Visualizing Spin Vectors...")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
    ax.set_facecolor('black')
    
    # Project 2D slice of DM
    mid = scaffold.params.grid_size // 2
    dm_slice = dm[:,:,mid]
    extent = [0, sim.box, 0, sim.box]
    
    im = ax.imshow(dm_slice.T, origin='lower', extent=extent, cmap='inferno', alpha=0.8)
    
    # Plot spin vectors for halos near the slice
    z_slice_thickness = 10.0 # physical units
    z_phys_mid = sim.box / 2
    
    count_plotted = 0
    for i, h_idx in enumerate(valid_halos):
        z_phys = h_idx[2] * (sim.box / scaffold.params.grid_size)
        
        # Only plot if near the slice
        if abs(z_phys - z_phys_mid) < z_slice_thickness:
            x = h_idx[0] * (sim.box / scaffold.params.grid_size)
            y = h_idx[1] * (sim.box / scaffold.params.grid_size)
            
            # Spin components projected (Lx, Ly)
            lx = spins[i][0]
            ly = spins[i][1]
            l_mag = np.sqrt(lx**2 + ly**2)
            
            if l_mag > 0:
                # Normalize length for plotting
                scale = 5.0 / (np.max(magnitudes) + 1e-10)
                ax.arrow(x, y, lx*scale, ly*scale, color='cyan', head_width=1.5, alpha=0.9)
                count_plotted += 1
                
    ax.set_title(f"Galactic Spin Generated by Dark Scaffold\n(Arrows = Angular Momentum Vectors)", 
                 color='white', fontsize=14)
    ax.set_xlabel("Mpc", color='white')
    ax.set_ylabel("Mpc", color='white')
    ax.tick_params(colors='white')
    
    plt.savefig('/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory/spin_test_result.png', 
               dpi=150, facecolor='black')
    print(f"   Plotted {count_plotted} halo spins on slice.")
    print("   Saved to spin_test_result.png")
    
    # Verdict
    print("\n" + "="*60)
    if avg_spin > 0.1:
        print("VERDICT: SIGNIFICANT SPIN DETECTED ✅")
        print("Matter falling into the asymmetric scaffold naturally spins up.")
    else:
        print("VERDICT: LOW SPIN DETECTED ⚠️")
        print("Scaffold might not be asymmetric enough or friction too high.")
    print("="*60)

if __name__ == "__main__":
    run_spin_test()
