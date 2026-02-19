"""
21cm Hydrogen Line Prediction
=============================

Objective: Predict the 21cm signal from the Dark Ages (z ~ 20-200) in the 
Dark Scaffold model.

Physics Background:
- Neutral hydrogen emits/absorbs at 21cm (1420 MHz rest frame).
- During the Dark Ages, there are no stars yet.
- The 21cm signal comes from hydrogen interacting with the CMB.
- In ΛCDM: Signal starts appearing around z ~ 30 as first structures form.
- In Dark Scaffold: Signal should appear EARLIER because structure pre-exists.

Prediction:
- Earlier structure → Earlier 21cm absorption
- Deeper absorption troughs due to pre-existing density contrasts

"""

import numpy as np
import matplotlib.pyplot as plt

def redshift_to_frequency(z):
    """Convert redshift to observed 21cm frequency."""
    nu_rest = 1420.0  # MHz
    return nu_rest / (1 + z)

def lcdm_21cm_signal(z):
    """
    Approximate ΛCDM 21cm brightness temperature.
    Based on Furlanetto et al. (2006) formalism.
    """
    # Before structure formation: signal is weak
    # Signal grows as structures form (z < 30)
    z_onset = 30.0  # When first structures start forming
    
    # Brightness temperature amplitude (simplified)
    T_max = -200.0  # mK (absorption against CMB)
    
    if z > 200:
        return 0.0  # Too early, gas not decoupled
    elif z > z_onset:
        # Dark ages: weak signal
        return -10.0 * (1 + z) / 100
    else:
        # Structure forming: deeper signal
        return T_max * np.exp(-(z - 20)**2 / 50)

def scaffold_21cm_signal(z):
    """
    Dark Scaffold 21cm brightness temperature.
    Key difference: Structure pre-exists.
    """
    # Structure exists from the beginning
    z_onset_scaffold = 100.0  # Much earlier onset!
    
    T_max = -250.0  # mK (deeper absorption due to pre-existing voids/filaments)
    
    if z > 200:
        return 0.0
    elif z > z_onset_scaffold:
        # Even in deep dark ages, scaffold creates density contrast
        return -20.0 * (1 + z) / 100
    else:
        # Pre-existing wells accelerate signal
        return T_max * np.exp(-(z - 50)**2 / 100)

def run_21cm_prediction():
    print("="*60)
    print("21cm HYDROGEN LINE PREDICTION")
    print("="*60)
    
    # Redshift range: Dark Ages
    z_range = np.linspace(10, 200, 200)
    
    # Calculate signals
    T_lcdm = np.array([lcdm_21cm_signal(z) for z in z_range])
    T_scaffold = np.array([scaffold_21cm_signal(z) for z in z_range])
    
    # Convert to observed frequency
    freq_range = redshift_to_frequency(z_range)
    
    # Find key differences
    max_diff_idx = np.argmax(np.abs(T_scaffold - T_lcdm))
    z_max_diff = z_range[max_diff_idx]
    freq_max_diff = freq_range[max_diff_idx]
    T_diff = T_scaffold[max_diff_idx] - T_lcdm[max_diff_idx]
    
    print("-" * 30)
    print("PREDICTIONS:")
    print(f"1. Maximum Signal Difference: {T_diff:.1f} mK at z = {z_max_diff:.0f}")
    print(f"   (Observed frequency: {freq_max_diff:.1f} MHz)")
    print(f"2. Scaffold onset: z ~ 100 (vs ΛCDM onset at z ~ 30)")
    print(f"3. Peak absorption: -250 mK (vs ΛCDM -200 mK)")
    print("-" * 30)
    
    # Observable with current experiments
    print("\n" + "="*60)
    print("TESTABILITY:")
    print("- HERA (Hydrogen Epoch of Reionization Array): Sensitive at 50-250 MHz")
    print("- SKA-Low: Planned sensitivity at 50-350 MHz")
    print(f"- Target frequency for maximum difference: {freq_max_diff:.1f} MHz")
    print("="*60)
    
    # Plot
    plt.figure(figsize=(12, 5), facecolor='black')
    
    ax1 = plt.subplot(1, 2, 1, facecolor='#1a1a2e')
    ax1.plot(z_range, T_lcdm, 'r--', linewidth=2, label='ΛCDM')
    ax1.plot(z_range, T_scaffold, 'c-', linewidth=2, label='Dark Scaffold')
    ax1.axvline(z_max_diff, color='yellow', linestyle=':', alpha=0.7, label=f'Max diff @ z={z_max_diff:.0f}')
    ax1.invert_xaxis()
    ax1.set_xlabel('Redshift z', color='white')
    ax1.set_ylabel('Brightness Temperature (mK)', color='white')
    ax1.set_title('21cm Signal: Dark Ages', color='white')
    ax1.legend()
    ax1.tick_params(colors='white')
    ax1.grid(alpha=0.2)
    
    ax2 = plt.subplot(1, 2, 2, facecolor='#1a1a2e')
    ax2.plot(freq_range, T_lcdm, 'r--', linewidth=2, label='ΛCDM')
    ax2.plot(freq_range, T_scaffold, 'c-', linewidth=2, label='Dark Scaffold')
    ax2.axvline(freq_max_diff, color='yellow', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Observed Frequency (MHz)', color='white')
    ax2.set_ylabel('Brightness Temperature (mK)', color='white')
    ax2.set_title('21cm Signal: Frequency Domain', color='white')
    ax2.legend()
    ax2.tick_params(colors='white')
    ax2.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('21cm_prediction.png', dpi=100, facecolor='black')
    print("   Saved plot to 21cm_prediction.png")
    
    return z_max_diff, freq_max_diff, T_diff

if __name__ == "__main__":
    run_21cm_prediction()
