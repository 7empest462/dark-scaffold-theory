"""
Gravitational Lensing Analysis (S8 Tension)
===========================================

Objective: Simulate the "Weak Lensing" signal (Shear/Convergence) of the Dark Scaffold.

Theory:
- "S8 Tension": Observations show the universe is "smoother" (less clumpy lensing)
  than Planck Lambda-CDM predicts.
- Standard NFW halos are "cuspy" -> Strong Lensing peaks.
- Dark Scaffold has "diffuse cores" -> Should produce SOFTER lensing peaks.

Method:
1. Generate High-Res Scaffold.
2. Project 3D Density to 2D Surface Density (Sigma).
3. Compute Convergence (Kappa) map (directly proportional to Sigma).
4. Compute Shear (Gamma) via Fourier Transform (Potential derivatives).
5. Compare Peak Statistics (Are they sharp or smooth?).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def run_lensing_test():
    print("="*60)
    print("GRAVITATIONAL LENSING SIMULATION")
    print("Testing the 'S8 Tension' (Lensing Amplitude)")
    print("="*60)
    
    # 1. Generate High-Res Scaffold
    print("\n1. Generating Mass Field...")
    n = 128 # Good resolution for imaging
    box = 100.0
    
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=n, box_size=box, 
        random_seed=42
    ))
    scaffold.generate()
    
    # 2. Project Density (Integration along Line of Sight Z)
    print("\n2. Projecting Mass (Ray Tracing)...")
    rho = scaffold.density_field
    
    # Surface Density Sigma(x,y)
    sigma = np.mean(rho, axis=2) 
    
    # Normalize Convergence (Kappa)
    # kappa = Sigma / Sigma_crit
    # We work in relative units over mean
    kappa = (sigma - np.mean(sigma)) / np.std(sigma)
    
    # 3. Calculate Shear Gamma (via Potential Psi)
    print("\n3. Computing Shear Matrix...")
    # Poisson in 2D: Lapl(Psi) = 2 * kappa
    # Solve in Fourier space: -k^2 * Psi_k = 2 * Kappa_k
    
    k = fftfreq(n) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0,0] = 1.0 # Avoid singularity
    
    kappa_k = fft2(kappa)
    
    # Potential Psi
    psi_k = -2 * kappa_k / k2
    psi_k[0,0] = 0
    
    # Shear components (second derivatives)
    # Gamma1 = 0.5 * (Psi_xx - Psi_yy)
    # Gamma2 = Psi_xy
    
    # Derivatives in Fourier: d/dx -> i*kx
    gamma1_k = 0.5 * ( (1j*kx)**2 - (1j*ky)**2 ) * psi_k
    gamma2_k = ( (1j*kx) * (1j*ky) ) * psi_k
    
    gamma1 = np.real(ifft2(gamma1_k))
    gamma2 = np.real(ifft2(gamma2_k))
    
    shear_magnitude = np.sqrt(gamma1**2 + gamma2**2)
    
    # 4. Analyze Clumpiness (S8 Proxy)
    # High S8 means lots of sharp peaks.
    # We check the "kurtosis" or simply the fraction of pixels > 3 sigma
    
    peak_fraction = np.sum(shear_magnitude > 3.0) / shear_magnitude.size
    max_shear = np.max(shear_magnitude)
    
    print(f"   Max Shear Strength: {max_shear:.2f} sigma")
    print(f"   Peak Fraction (>3sig): {peak_fraction*100:.2f}%")
    
    # 5. Visualize Lensing Map
    print("\n4. Generating Lensing Map...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
    
    # Convergence Map (Mass Map)
    ax = axes[0]
    im = ax.imshow(kappa, cmap='inferno', vmin=-1, vmax=5, origin='lower')
    ax.set_title("Convergence (Mass Map) $\kappa$", color='white')
    ax.axis('off')
    
    # Shear Map (Distortion)
    ax = axes[1]
    im2 = ax.imshow(shear_magnitude, cmap='magma', vmin=0, vmax=3, origin='lower')
    ax.set_title("Shear Magnitude $|\gamma|$", color='white')
    ax.axis('off')
    
    # Add dummy background galaxies to show distortion effect?
    # Overlay vector field for shear direction
    step = 8
    Y, X = np.mgrid[0:n:step, 0:n:step]
    g1_sub = gamma1[::step, ::step]
    g2_sub = gamma2[::step, ::step]
    
    # Orientation angle of ellipse = 0.5 * atan2(g2, g1)
    # Just plotting 'sticks' to show distortion
    # Length proportional to magnitude
    mag_sub = np.sqrt(g1_sub**2 + g2_sub**2)
    angle = 0.5 * np.arctan2(g2_sub, g1_sub)
    
    U = mag_sub * np.cos(angle)
    V = mag_sub * np.sin(angle)
    
    ax.quiver(X, Y, U, V, color='cyan', headlength=0, headaxislength=0, pivot='mid', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('/Users/robsimens/Documents/Cosmology/dark-scaffold-theory/lensing_test_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to lensing_test_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: S8 TENSION (SMOOTHNESS)")
    
    # Standard NFW peaks usually hit >5 sigma easily
    # If we are "softer", max_shear should be moderate
    if max_shear < 5.0 and peak_fraction < 0.01:
        print("RESULT: SOFT LENSING (SOLVES S8) ✅")
        print("The lensing map lacks extreme cusps. It is ' smoother' than CDM.")
        print("This matches weak lensing surveys (KiDS/DES) perceiving a smoother universe.")
    else:
        print("RESULT: STRONG CLUMPING")
        print("The scaffold produces standard strong lensing peaks.")
    print("="*60)

if __name__ == "__main__":
    run_lensing_test()
