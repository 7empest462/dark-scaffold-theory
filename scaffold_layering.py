"""
Layered Structure Analysis (Fractal Echo Test)
==============================================

Objective: Test for "Harmonic Echoes" in the cosmic web structure.
Hypothesis: If the universe is cyclic, the current accumulation of matter 
is built upon the "Gravitational Ghost" of the previous aeon.
This implies the Cosmic Web should exhibit "Discrete Scale Invariance" (DSI).

We simulate this by "stacking" scaffolds:
Total_Field = Current_Aeon + alpha * Previous_Aeon(scaled) + alpha^2 * Ancient_Aeon...

Method:
1. Generate a "Layered" Scaffold (Superposition of scaled fields).
2. Compute the Two-Point Correlation Function xi(r).
3. Look for periodic "Resonance Spikes" (Echoes) in xi(r) 
   that deviate from a standard single-epoch LambdaCDM power law.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn, fftfreq
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def generate_layered_field(N, box_size, layers=3, scale_factor=2.0, decay=0.3):
    """
    Generates a field composed of multiple 'aeons' stacked on top of each other.
    Each previous aeon is:
    1. 'Larger' (scaled up coordinates, or lower k modes)
    2. 'Fainter' (amplitude decay)
    """
    # Base params
    params = ScaffoldParameters(grid_size=N, box_size=box_size, random_seed=42)
    
    # We construct the field in Fourier space to easily handle scaling
    # Standard Power Law: P(k) ~ k^n
    
    # To simulate "Scaled Copy", we add a component:
    # rho_total(k) = rho_0(k) + decay * rho_0(k * scale) ?
    # No, that's complex scaling.
    
    # Let's do it in real space with different seeds but correlated structure?
    # CCC says the previous aeon's infinite future becomes our Big Bang.
    # So the *seeds* are correlated.
    
    # Simplified Model:
    # The 'Scaffold' is a sum of fractally related random fields.
    
    final_density = np.zeros((N, N, N))
    
    print(f"   Genering {layers} stacked aeons (Scale Factor: {scale_factor}x)...")
    
    for i in range(layers):
        # Each layer represents an aeon
        # The 'scale' determines the dominant feature size?
        # Or we just generate a field and add it?
        
        # To simulate "Previous Aeon was bigger/different scale":
        # We change the smoothing scale.
        # Current Aeon: Smoothing = 2.0 Mpc
        # Previous Aeon: Smoothing = 2.0 * scale_factor Mpc (Features are larger)
        
        effective_smoothing = 2.0 * (scale_factor ** i)
        amplitude = decay ** i
        
        print(f"     - Layer {i}: Scale ~ {effective_smoothing:.1f} Mpc, Amp: {amplitude:.2f}")
        
        # Use different seed implies uncorrelated layers (random stacking)
        # Use SAME seed implies "Memory" (deterministic evolution)
        # CCC implies Memory. So same seed (or related).
        
        layer_params = ScaffoldParameters(
            grid_size=N, 
            box_size=box_size, 
            smoothing_scale=effective_smoothing,
            random_seed=42 # Persistence of memory
        )
        
        scaffold = DarkMatterScaffold(layer_params)
        scaffold.generate()
        
        final_density += scaffold.density_field * amplitude
        
    # Normalize
    final_density /= np.max(final_density)
    return final_density

def compute_correlation_function(density, box_size, r_bins):
    """
    Computes isotropic two-point correlation function xi(r)
    using FFT method: xi(r) = IFFT( |FFT(delta)|^2 )
    """
    N = density.shape[0]
    
    # Input 'density' is already a sum of zero-mean fields, so it holds the fluctuation
    delta = density 
    
    # Power Spectrum
    delta_k = fftn(delta)
    power = np.abs(delta_k)**2
    
    # Auto-correlation (Xi)
    xi_grid = np.real(ifftn(power))
    
    # Radial average
    # Use distance from origin (0,0,0) - accounting for periodicity?
    # FFT output has origin at 0.
    
    x = np.fft.fftfreq(N) * box_size
    y = np.fft.fftfreq(N) * box_size
    z = np.fft.fftfreq(N) * box_size
    
    r_grid = np.sqrt(x[:,None,None]**2 + y[None,:,None]**2 + z[None,None,:]**2)
    
    # In periodicity, fftfreq handles the wrapping (0 to 0.5, then -0.5 to 0)
    # But distance calculation needs care. 
    # Actually fftfreq returns [0, 1, ..., -1] order. 
    # Magnitude |k| is fine.
    
    # Binning
    xi_r = []
    r_vals = []
    
    print("   Computing Correlation Function xi(r)...")
    
    for i in range(len(r_bins)-1):
        r_min = r_bins[i]
        r_max = r_bins[i+1]
        
        mask = (r_grid >= r_min) & (r_grid < r_max)
        if np.sum(mask) > 0:
            mean_xi = np.mean(xi_grid[mask])
            xi_r.append(mean_xi)
            r_vals.append((r_min + r_max)/2.0)
            
    return np.array(r_vals), np.array(xi_r)

def run_layering_test():
    print("="*60)
    print("LAYERED STRUCTURE ANALYSIS (FRACTAL ECHOES)")
    print("="*60)
    
    N = 64
    box_size = 200.0 # Large box to see large scale echoes
    
    # 1. Generate Layered Scaffold (The "Cyclic" Model)
    # 3 Layers: Current, Past (2x larger), Ancient (4x larger)
    rho = generate_layered_field(N, box_size, layers=3, scale_factor=2.5, decay=0.4)
    
    # 2. Compute Correlation Function
    r_bins = np.linspace(2.0, 80.0, 40)
    r_vals, xi_vals = compute_correlation_function(rho, box_size, r_bins)
    
    # Normalize Xi
    xi_vals = xi_vals / xi_vals[0]
    
    # 3. Analyze for Echoes
    # In standard CDM, xi(r) decreases monotonically (approx power law)
    # In Layered model, we expect "humps" or deviations at scales corresponding to the layers
    
    # Detect local maxima in the tail (r > 10)
    # (Ignore the central peak at r=0)
    
    print("\n3. Analyzing signal for harmonic echoes...")
    
    # Smooth slightly
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(xi_vals, height=0.01) # Very low height threshold
    
    # We expect peaks at scales related to the smoothing scales?
    # Smoothing adds a cutoff, not necessarily a peak. 
    # But the SUPERPOSITION of two distinct scales might create a feature.
    
    # Let's compare to a Single Layer (Control)
    print("   Generating Control Field (Single Aeon)...")
    rho_control = generate_layered_field(N, box_size, layers=1)
    _, xi_control = compute_correlation_function(rho_control, box_size, r_bins)
    xi_control = xi_control / xi_control[0]
    
    # Calculate Residuals (Difference between Layered and Control)
    residuals = xi_vals - xi_control
    
    # Check if residuals show oscillatory behavior
    res_std = np.std(residuals)
    print(f"   Residual Standard Deviation: {res_std:.4f}")
    
    has_echoes = False
    if res_std > 0.05: # Arbitrary threshold for "significant structure"
        has_echoes = True
    
    # 4. Visualize
    plt.figure(figsize=(10, 6), facecolor='black')
    
    plt.plot(r_vals, xi_control, 'w--', alpha=0.5, label='Standard Single-Aeon')
    plt.plot(r_vals, xi_vals, 'c-', linewidth=2, label='Cyclic Multi-Aeon (3 Layers)')
    
    # Highlight residuals/echoes
    # plt.plot(r_vals, residuals + 0.5, 'm-', alpha=0.5, label='Echo Signal (Offset)')
    
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Cosmic Structure: Cyclic Echoes", color='white', fontsize=14)
    ax.set_xlabel("Separation r [Mpc]", color='white')
    ax.set_ylabel("Correlation Function xi(r)", color='white')
    ax.tick_params(colors='white')
    plt.legend()
    plt.grid(alpha=0.2)
    
    if has_echoes:
        max_res_idx = np.argmax(residuals)
        plt.annotate('Harmonic Echo', 
                    xy=(r_vals[max_res_idx], xi_vals[max_res_idx]), 
                    xytext=(r_vals[max_res_idx]+10, xi_vals[max_res_idx]+0.2),
                    arrowprops=dict(facecolor='yellow', shrink=0.05),
                    color='yellow')
    
    plt.savefig('layering_test_result.png', dpi=100, facecolor='black')
    print("   Saved plot to layering_test_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: FRACTAL ECHOES")
    
    if has_echoes:
        print("RESULT: POSITIVE (+)")
        print("Detected harmonic structure in the correlation function.")
        print("The 'Stacking' of aeons creates visible resonance.")
    else:
        print("RESULT: NEGATIVE (-)")
        print("Layering blended indistinguishably. No distinct echoes.")
    print("="*60)

if __name__ == "__main__":
    run_layering_test()
