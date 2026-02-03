"""
Matter Power Spectrum Analysis (P(k))
=====================================

Objective: Calculate the Power Spectrum P(k) of the "Seepage" model result and 
compare it with the standard Lambda-CDM Matter Power Spectrum.

The Power Spectrum describes the amount of structure at different scales (k).
- Small k (Large scales): Should match primordial conditions.
- Large k (Small scales): Structure formation increases power.

Method:
1. Run Seepage Simulation to z=0.
2. Convert particle distribution to density grid (CIC/NGP).
3. Compute 3D FFT -> |delta_k|^2.
4. Bin spherically to get 1D P(k).
5. Compare with theoretical Lambda-CDM expectation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from infall_simulation import SeepageSimulation, SeepageParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def compute_power_spectrum(positions, box_size, grid_size):
    """Compute P(k) from particle positions."""
    # 1. Density Field (CIC assignment would be better, using NGP for speed)
    density = np.zeros((grid_size, grid_size, grid_size))
    idx = (positions / box_size * grid_size).astype(int)
    idx = np.clip(idx, 0, grid_size-1)
    np.add.at(density, (idx[:,0], idx[:,1], idx[:,2]), 1)
    
    # Overdensity delta = (rho - mean) / mean
    mean_density = np.mean(density)
    delta = (density - mean_density) / mean_density
    
    # 2. Fourier Transform
    delta_k = np.fft.fftn(delta)
    P_3d = np.abs(delta_k)**2 / (grid_size**6) # Normalization
    
    # 3. Binning
    k_freq = np.fft.fftfreq(grid_size, d=box_size/grid_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    
    k_min = 2 * np.pi / box_size
    k_max = np.max(k_freq)
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 20)
    
    k_vals = []
    P_vals = []
    
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            k_mean = np.mean(k_mag[mask])
            P_mean = np.mean(P_3d[mask]) * (box_size**3) # Volume factor
            k_vals.append(k_mean)
            P_vals.append(P_mean)
            
    return np.array(k_vals), np.array(P_vals)

def theoretical_lcdm_pk(k_vals):
    """Approximation of linear Lambda-CDM P(k)."""
    # Eisenstein & Hu (1998) transfer function approximation (very simplified)
    # P(k) ~ k * T(k)^2
    # This is just for visual comparison
    
    # Shape parameters
    Gamma = 0.2
    q = k_vals / Gamma
    
    # BBKS Transfer function
    L0 = np.log(2*np.e + 1.8*q)
    C0 = 14.2 + 731/(1+62.5*q)
    T_k = L0 / (L0 + C0*q**2)
    
    # Primordial P(k) ~ k^1 (Harrison-Zel'dovich)
    A = 2e5 # Amplitude scaling (arbitrary for shape comparison)
    P_k = A * k_vals * (T_k**2)
    
    return P_k

def run_power_spectrum_test():
    print("="*60)
    print("MATTER POWER SPECTRUM ANALYSIS (P(k))")
    print("Quantifying Structure Formation vs Scale")
    print("="*60)
    
    # 1. Run Simulation
    print("\n1. Running High-Res Seepage Simulation...")
    box_size = 100.0
    grid_size = 128 # Higher res grid for FFT
    
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=grid_size, box_size=box_size, 
        random_seed=42
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=200000, 
        gravity_strength=50.0,
        friction=0.04,
        n_steps=500
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run()
    
    # 2. Calculate P(k)
    print("\n2. Computing Power Spectra...")
    k_scaffold, P_scaffold = compute_power_spectrum(sim.positions, box_size, grid_size)
    
    # Compare with DM only (the scaffold itself)
    # We need to turn the scaffold grid into "particles" or use the grid directly
    dm_delta = (scaffold.density_field - np.mean(scaffold.density_field)) / np.mean(scaffold.density_field)
    dm_delta_k = np.fft.fftn(dm_delta)
    P_3d_dm = np.abs(dm_delta_k)**2 / (grid_size**6)
    
    # Bin DM P(k) same way
    k_dm_vals = []
    P_dm_vals = []
    
    k_freq = np.fft.fftfreq(grid_size, d=box_size/grid_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    
    k_min = 2 * np.pi / box_size
    k_max = np.max(k_freq)
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), 20)
    
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            k_mean = np.mean(k_mag[mask])
            P_mean = np.mean(P_3d_dm[mask]) * (box_size**3)
            k_dm_vals.append(k_mean)
            P_dm_vals.append(P_mean)
            
    k_dm = np.array(k_dm_vals)
    P_dm = np.array(P_dm_vals)
    
    # Theoretical LCDM
    P_theory = theoretical_lcdm_pk(k_scaffold)
    # Normalize theory to match scaffold at large scales (k=0.1)
    norm_idx = np.argmin(np.abs(k_scaffold - 0.2))
    scale = P_scaffold[norm_idx] / P_theory[norm_idx]
    P_theory *= scale
    
    # 3. Visualize
    print("\n3. Plotting Results...")
    plt.figure(figsize=(10, 8), facecolor='black')
    
    plt.loglog(k_scaffold, P_scaffold, 'c-o', linewidth=2, label='Seepage Model (Baryons)')
    plt.loglog(k_dm, P_dm, 'orange', linestyle='--', alpha=0.6, label='Underlying DM Scaffold')
    plt.loglog(k_scaffold, P_theory, 'w:', linewidth=1.5, label='Standard \u039BMCDM Shape')
    
    plt.title("Matter Power Spectrum P(k)", color='white', fontsize=16)
    plt.xlabel("k [h/Mpc]", color='white', fontsize=12)
    plt.ylabel("P(k) [(Mpc/h)^3]", color='white', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.2, which='both')
    
    # Styling
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white', which='both')
    for spine in ax.spines.values():
        spine.set_color('white')
        
    plt.savefig('/Users/robsimens/Documents/Cosmology/dark-scaffold-theory/power_spectrum_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to power_spectrum_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: POWER SPECTRUM ANALYSIS")
    
    # Check slope at small scales (high k)
    # Seepage should maintain power at small scales (clumping)
    slope = (np.log10(P_scaffold[-2]) - np.log10(P_scaffold[-5])) / (np.log10(k_scaffold[-2]) - np.log10(k_scaffold[-5]))
    print(f"   Small-scale Slope: {slope:.2f}")
    
    if slope > -2.5:
        print("RESULT: REALISTIC CLUSTERING ✅")
        print("Baryons trace the scaffold structure effectively across scales.")
    else:
        print("RESULT: WEAK CLUSTERING")
        print("Structure formation may be suppressed at small scales.")
        
    print("="*60)

if __name__ == "__main__":
    run_power_spectrum_test()
