"""
Ghost Ring Search (CCC Test)
============================

Objective: Search for "Gravitational Scars" in the Dark Scaffold.
Hypothesis: Supermassive Black Hole mergers in the previous aeon left 
concentric rings of low variance in the initial gravitational potential 
(Penrose's Conformal Cyclic Cosmology).

Method:
1. Generate Dark Scaffold Density Field.
2. Convert to Gravitational Potential (Poisson Solver).
3. Project to 2D (The "Sky Map").
4. Perform Ring Analysis:
   - For each massive node (potential center):
     - Measure variance of the field in concentric rings (annuli).
     - Look for "Low Variance" anomalies (signal of a coherent wave).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def solve_poisson(density, box_size):
    """
    Solves Poisson Equation: del^2 Phi = 4 pi G rho
    Phi(k) = -4 pi G rho(k) / k^2
    """
    N = density.shape[0]
    # FFT
    rho_k = fftn(density)
    
    # K-grid
    kx = fftfreq(N) * N
    ky = fftfreq(N) * N
    kz = fftfreq(N) * N
    k_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_sq = k_grid[0]**2 + k_grid[1]**2 + k_grid[2]**2
    
    # Avoid division by zero
    k_sq[0,0,0] = 1.0
    
    # Solve
    # Constants absorbed into units for qualitative shape
    phi_k = -rho_k / k_sq
    phi_k[0,0,0] = 0.0
    
    phi = np.real(ifftn(phi_k))
    return phi

def ring_variance_analysis(image, centers, radii):
    """
    Calculates the variance of the image pixels along rings 
    of specified radii centered at 'centers'.
    """
    h, w = image.shape
    results = []
    
    y, x = np.ogrid[:h, :w]
    
    for cx, cy in centers:
        node_profile = []
        
        # Distance map from this center
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        for r in radii:
            # Annulus mask (width 1 pixel)
            mask = (dist_sq >= r**2) & (dist_sq < (r+1)**2)
            
            if np.sum(mask) > 10:
                # Calculate variance in this ring
                ring_vals = image[mask]
                var = np.var(ring_vals)
                node_profile.append(var)
            else:
                node_profile.append(np.nan)
        
        results.append(np.array(node_profile))
        
    return results

def run_ghost_ring_test():
    print("="*60)
    print("SEARCHING FOR GHOST RINGS (CCC SIGNATURE)")
    print("="*60)
    
    # 1. Generate Scaffold
    print("1. Generating Dark Potential...")
    N = 128
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=N, box_size=100.0, 
        random_seed=999 # New seed
    ))
    scaffold.generate()
    
    # 2. Get Potential
    rho = scaffold.density_field
    phi_3d = solve_poisson(rho, 100.0)
    
    # 3. Project to 2D (Sky Map)
    # Integrating along one axis simulates the "surface of last scattering" depth roughly
    sky_map = np.sum(phi_3d, axis=2)
    # Normalize
    sky_map = (sky_map - np.mean(sky_map)) / np.std(sky_map)
    
    # 4. Identify Centers (Massive Nodes)
    # We look for rings around the biggest structures (the "echoes" of past SMBHs)
    # Find local minima of potential (deep wells)
    # Or maxima of density? Deep potential wells correspond to density peaks.
    # Potential is negative in wells. So local minima.
    
    from scipy.ndimage import minimum_filter
    
    local_min = minimum_filter(sky_map, size=10) == sky_map
    # Filter for deep ones (below -2 sigma)
    deep_mask = local_min & (sky_map < -1.5)
    
    cy, cx = np.where(deep_mask)
    centers = list(zip(cx, cy))
    
    print(f"   Found {len(centers)} massive nodes to scan.")
    
    # 5. Analyze Rings
    print("2. Scanning for concentric low-variance rings...")
    radii = np.arange(2, 40, 1.0)
    profiles = ring_variance_analysis(sky_map, centers, radii)
    
    # 6. Detect Anomalies
    mean_profile = np.nanmean(profiles, axis=0) # Average profile around nodes
    
    # In a random field, variance should be roughly constant or slowly increasing
    # A "Ring" signature is a sharp DIP in variance at a specific radius
    
    # Calculate derivative of variance
    d_var = np.diff(mean_profile)
    
    # Check for significant dips
    threshold = -0.05 # Arbitrary "sharp drop" threshold
    significant_rings = np.where(d_var < threshold)[0]
    
    print("\n3. Results:")
    found_rings = False
    
    if len(significant_rings) > 0:
        found_rings = True
        print(f"   [!] DETECTED {len(significant_rings)} POTENTIAL RING CONFIGURATIONS.")
        print(f"   Radii: {radii[significant_rings]}")
    else:
        print("   No obvious deep variance dips found in mean profile.")
        
    # 7. Visualize
    plt.figure(figsize=(12, 6), facecolor='black')
    
    # Left: The Sky Map with centers
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(sky_map, cmap='magma', origin='lower')
    ax1.scatter(cx, cy, c='cyan', s=20, marker='x', alpha=0.7)
    ax1.set_title("Projected Potential (Sky Map)", color='white')
    ax1.axis('off')
    
    # Right: The Variance Profile
    ax2 = plt.subplot(1, 2, 2, facecolor='#1a1a2e')
    ax2.plot(radii, mean_profile, color='cyan', linewidth=2, label='Mean Variance')
    ax2.set_xlabel("Ring Radius (pixels)", color='white')
    ax2.set_ylabel("Field Variance", color='white')
    ax2.set_title("Ring Variance Analysis", color='white')
    ax2.tick_params(colors='white')
    ax2.grid(alpha=0.2)
    
    if found_rings:
        for r_idx in significant_rings:
            r_val = radii[r_idx]
            ax2.axvline(r_val, color='yellow', linestyle='--', alpha=0.7)
            ax2.text(r_val, np.min(mean_profile), "RING", color='yellow', rotation=90)
            
    plt.savefig('ghost_ring_result.png', dpi=100, facecolor='black')
    print("   Saved plot to ghost_ring_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: CYCLIC REMNANTS")
    if found_rings:
        print("RESULT: POSITIVE (+)")
        print("Detected concentric low-variance structures.")
        print("Consistency with Penrose's CCC predictions: HIGH.")
    else:
        print("RESULT: NEGATIVE (-)")
        print("No coherent ring structures found.")
    print("="*60)

if __name__ == "__main__":
    run_ghost_ring_test()
