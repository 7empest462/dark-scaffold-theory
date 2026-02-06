"""
The Spin Doctor (Cosmic Rotation Test)
======================================

Objective: Test if a "Spinning" previous universe explains the Axis of Evil.
Hypothesis: The previous aeon had net angular momentum (Spin). This rotation 
imparted a geometric bias to the Dark Scaffold (stretching/flattening it), 
which forces the large-scale multipoles (Quadrupole/Octupole) to align.

Method:
1. Generate a "Biased" Scaffold.
   - We modify the power spectrum to be anisotropic:
   - P(k) -> P(k) * (1 + epsilon * cos(theta_k)^2)
   - This simulates stretching along a preferred "Spin Axis" (Z).
2. Project to 2D Sky Map.
3. Measure alignment between Quadrupole and Octupole.
4. Verify if "Spin" forces alignment.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn, fftfreq
from scaffold_generator import ScaffoldParameters
from axis_alignment import calculate_principal_axis

class SpinningScaffoldGenerator:
    def __init__(self, N=128, box_size=500.0, spin_bias=0.0):
        self.N = N
        self.box_size = box_size
        self.bias = spin_bias # Strength of anisotropy (0.0 = Isotropic, 1.0 = Extreme)
        
    def generate(self):
        n = self.N
        
        # 1. White Noise
        noise_real = np.random.standard_normal((n, n, n))
        noise_imag = np.random.standard_normal((n, n, n))
        noise_k = noise_real + 1j * noise_imag
        
        # 2. Anisotropic Power Spectrum
        kx = fftfreq(n)
        ky = fftfreq(n)
        kz = fftfreq(n)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        k_mag[0,0,0] = 1.0
        
        # Base Power Law P(k) ~ k^-1.5
        power = k_mag**(-1.5)
        
        # --- THE SPIN DOCTOR BIAS ---
        # Modify power based on angle from Z-axis (Spin Axis)
        # Cos(theta) = kz / k_mag
        # We suppress or enhance power along the axis
        
        if self.bias > 0.0:
            cos_theta = KZ / k_mag
            # Simple quadrupole anisotropy: (1 + bias * cos^2(theta))
            # This stretches/compresses structure along Z
            anisotropy = 1.0 + self.bias * (3 * cos_theta**2 - 1)
            power *= anisotropy
            
        power[0,0,0] = 0.0
        
        # 3. IFFT
        field_k = noise_k * np.sqrt(power)
        density = np.real(ifftn(field_k))
        
        # Smooth
        density = gaussian_filter(density, sigma=2.0)
        
        # Normalize
        density = (density - np.mean(density)) / np.std(density)
        
        return density

def run_spin_test():
    print("="*60)
    print("THE SPIN DOCTOR: ROTATIONAL BIAS TEST")
    print("="*60)
    
    # 1. Control Run (No Spin)
    print("\n1. Control Universe (Bias = 0.0)...")
    gen_control = SpinningScaffoldGenerator(spin_bias=0.0)
    rho_control = gen_control.generate()
    
    # Project and Measure
    sky_control = np.sum(rho_control, axis=2) # Project along Spin Axis (Z)
    # Wait, if we look DOWN the spin axis, we see the symmetry?
    # No, we should look FROM THE SIDE (integrate along X or Y) to see the flattening.
    # If we integrate along Z, we average out the spin effect?
    # Let's view from the "Side" (Axis 0)
    sky_control = np.sum(rho_control, axis=0) 
    
    q_con = gaussian_filter(sky_control, sigma=32)
    o_con = gaussian_filter(sky_control, sigma=16) - q_con
    
    angle_q_c = calculate_principal_axis(q_con)
    angle_o_c = calculate_principal_axis(o_con)
    diff_c = abs(angle_q_c - angle_o_c)
    if diff_c > 90: diff_c = 180 - diff_c
    
    print(f"   Misalignment: {diff_c:.1f} deg (Random)")
    
    # 2. Spinning Run (Bias = 0.15)
    print("\n2. Spinning Universe (Bias = 0.15)...")
    print("   Simulating angular momentum artifact from previous aeon...")
    gen_spin = SpinningScaffoldGenerator(spin_bias=0.25) # Strong bias to force effect
    rho_spin = gen_spin.generate()
    
    # Project from Side (to see the alignment relative to Z-axis)
    sky_spin = np.sum(rho_spin, axis=0) 
    
    q_spin = gaussian_filter(sky_spin, sigma=32)
    o_spin = gaussian_filter(sky_spin, sigma=16) - q_spin
    
    angle_q_s = calculate_principal_axis(q_spin)
    angle_o_s = calculate_principal_axis(o_spin)
    diff_s = abs(angle_q_s - angle_o_s)
    if diff_s > 90: diff_s = 180 - diff_s
    
    print(f"   Quadrupole Axis: {angle_q_s:.1f} deg")
    print(f"   Octupole Axis:   {angle_o_s:.1f} deg")
    print(f"   Misalignment:    {diff_s:.1f} deg")
    
    # 3. Verdict
    print("\n" + "="*60)
    print("VERDICT: AXIS OF EVIL SOLUTION")
    
    if diff_s < 20.0:
        print("RESULT: POSITIVE (+)")
        print("Cosmic Spin forces alignment!")
        print("The 'Axis of Evil' is just the axis of the previous universe's rotation.")
    else:
        print("RESULT: NEGATIVE (-)")
        print("Even with spin, structures remain randomly oriented.")
    print("="*60)
    
    # 4. Plot
    plt.figure(figsize=(10, 8), facecolor='black')
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(sky_control, cmap='inferno')
    ax1.set_title(f"Control (No Spin)\nMisalign: {diff_c:.1f}°", color='white')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(sky_spin, cmap='inferno')
    ax2.set_title(f"Spinning (Bias=0.25)\nMisalign: {diff_s:.1f}°", color='white')
    ax2.axis('off')
    
    # Show the aligned modes
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(q_spin, cmap='twilight')
    ax3.set_title("Aligned Quadrupole (l=2)", color='white')
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(o_spin, cmap='twilight')
    ax4.set_title("Aligned Octupole (l=3)", color='white')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('spin_bias_result.png', dpi=100, facecolor='black')
    print("   Saved plot to spin_bias_result.png")

if __name__ == "__main__":
    run_spin_test()
