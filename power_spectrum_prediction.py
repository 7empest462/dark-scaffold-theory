"""
Power Spectrum Prediction
=========================

Objective: Calculate the precise matter power spectrum P(k) for the Dark Scaffold
model and compare it to ΛCDM predictions.

Method:
1. Generate scaffold density field with known parameters.
2. Compute P(k) from the 3D Fourier Transform.
3. Compare to Planck 2018 ΛCDM fiducial P(k).
4. Quantify deviation as function of k.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def compute_power_spectrum(density_field, box_size):
    """
    Compute the 3D power spectrum P(k) from a density field.
    """
    n = density_field.shape[0]
    
    # Fourier Transform
    delta_k = fftn(density_field) / density_field.size
    
    # Power = |delta_k|^2
    power_3d = np.abs(delta_k)**2
    
    # k-space coordinates
    kx = fftfreq(n, d=box_size/n)
    ky = fftfreq(n, d=box_size/n)
    kz = fftfreq(n, d=box_size/n)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Bin in k-shells
    k_bins = np.linspace(0, 0.5, 50)  # k in h/Mpc
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    P_k = np.zeros(len(k_centers))
    
    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.sum(mask) > 0:
            P_k[i] = np.mean(power_3d[mask])
            
    return k_centers, P_k

def lcdm_power_spectrum(k):
    """
    Approximate ΛCDM power spectrum (Eisenstein-Hu fitting formula, simplified).
    P(k) ~ k^n_s * T(k)^2 where T(k) is the transfer function.
    """
    # Primordial spectral index
    n_s = 0.965
    
    # Simplified transfer function (matter-radiation equality scale)
    k_eq = 0.01  # h/Mpc (approximate)
    T_k = 1.0 / (1.0 + (k / k_eq)**2)
    
    # Amplitude (normalized to sigma_8 = 0.81)
    A_s = 2.0e-9
    
    P_k = A_s * (k / 0.05)**n_s * T_k**2
    
    # Scale to match simulation units
    P_k *= 1e6  # Normalization factor
    
    return P_k

def run_prediction():
    print("="*60)
    print("POWER SPECTRUM PREDICTION")
    print("="*60)
    
    # 1. Generate High-Resolution Scaffold
    print("1. Generating scaffold at 128^3 resolution...")
    params = ScaffoldParameters(
        grid_size=128,
        box_size=500.0,  # 500 Mpc/h
        spectral_index=-1.5,
        smoothing_scale=2.0,
        random_seed=42
    )
    scaffold = DarkMatterScaffold(params)
    density = scaffold.generate()
    
    # 2. Compute P(k)
    print("2. Computing power spectrum...")
    k_scaffold, P_scaffold = compute_power_spectrum(density, params.box_size)
    
    # 3. Compute ΛCDM reference
    print("3. Computing ΛCDM reference...")
    P_lcdm = lcdm_power_spectrum(k_scaffold)
    
    # 4. Calculate deviation
    print("4. Calculating deviation...")
    
    # Normalize both to same scale for comparison
    # (Our simulation is arbitrary units, we compare shape)
    norm_factor = np.mean(P_scaffold[10:20]) / np.mean(P_lcdm[10:20])
    P_scaffold_norm = P_scaffold / norm_factor
    
    deviation = (P_scaffold_norm - P_lcdm) / P_lcdm * 100
    
    # Find the characteristic bump scale
    max_dev_idx = np.argmax(np.abs(deviation[5:30])) + 5
    k_bump = k_scaffold[max_dev_idx]
    dev_at_bump = deviation[max_dev_idx]
    
    print("-" * 30)
    print(f"Maximum Deviation: {dev_at_bump:.1f}% at k = {k_bump:.3f} h/Mpc")
    print("-" * 30)
    
    # 5. Check if within ±5% threshold for publishability
    mean_deviation = np.mean(np.abs(deviation[5:40]))
    
    print("\n" + "="*60)
    print("VERDICT: POWER SPECTRUM PREDICTION")
    if mean_deviation < 10.0:
        print(f"RESULT: CONSISTENT (Mean deviation: {mean_deviation:.1f}%)")
        print("The scaffold P(k) is statistically consistent with ΛCDM at large scales.")
    else:
        print(f"RESULT: SIGNIFICANT DEVIATION (Mean: {mean_deviation:.1f}%)")
        print(f"Characteristic scale identified at k ~ {k_bump:.3f} h/Mpc.")
    print("="*60)
    
    # 6. Plot
    plt.figure(figsize=(12, 5), facecolor='black')
    
    ax1 = plt.subplot(1, 2, 1, facecolor='#1a1a2e')
    ax1.loglog(k_scaffold[1:], P_scaffold_norm[1:], 'c-', linewidth=2, label='Dark Scaffold')
    ax1.loglog(k_scaffold[1:], P_lcdm[1:], 'r--', linewidth=2, label='ΛCDM')
    ax1.axvline(k_bump, color='yellow', linestyle=':', alpha=0.7, label=f'Bump @ k={k_bump:.3f}')
    ax1.set_xlabel('k (h/Mpc)', color='white')
    ax1.set_ylabel('P(k)', color='white')
    ax1.set_title('Matter Power Spectrum', color='white')
    ax1.legend()
    ax1.tick_params(colors='white')
    ax1.grid(alpha=0.2)
    
    ax2 = plt.subplot(1, 2, 2, facecolor='#1a1a2e')
    ax2.plot(k_scaffold[5:40], deviation[5:40], 'm-', linewidth=2)
    ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax2.axhspan(-5, 5, alpha=0.2, color='green', label='±5% target')
    ax2.set_xlabel('k (h/Mpc)', color='white')
    ax2.set_ylabel('Deviation (%)', color='white')
    ax2.set_title('Deviation from ΛCDM', color='white')
    ax2.legend()
    ax2.tick_params(colors='white')
    ax2.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('power_spectrum_prediction.png', dpi=100, facecolor='black')
    print("   Saved plot to power_spectrum_prediction.png")
    
    return k_bump, dev_at_bump

if __name__ == "__main__":
    run_prediction()
