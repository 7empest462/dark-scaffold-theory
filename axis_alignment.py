"""
Cosmic Alignment Test ("Axis of Evil")
======================================

Objective: Test if the Dark Scaffold breaks isotropy and explains the 
observed alignment of the Quadrupole (l=2) and Octupole (l=3) modes in the CMB.

Hypothesis: If the Scaffold is a remnant, it might carry a preferred direction 
(e.g., the rotation axis of the previous universe).
Standard ΛCDM predicts random orientations (Isotropy).
Observations show they are aligned (The Axis of Evil).

Method:
1. Generate Dark Scaffold Potential.
2. Project to 2D Sky Map.
3. Decompose into Spherical Harmonics (or 2D Multipoles).
   - We use simple 2D Moment Analysis for the projected map.
   - Moment of Inertia Tensor -> Principal Axes.
4. Compare Principal Axes of Large Scale Structures (Quadrupole) vs slightly smaller (Octupole).
5. Measure Angle between axes.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def calculate_principal_axis(image, threshold_percent=20):
    """
    Calculates the principal axis (orientation angle) of the structures 
    in the image using the Moment of Inertia tensor.
    """
    # Threshold to identify "Structures"
    # We focus on the high-density regions or low-potential wells
    binary_mask = image > np.percentile(image, 100 - threshold_percent)
    
    y, x = np.where(binary_mask)
    weights = image[binary_mask]
    
    # Center of Mass
    com_x = np.average(x, weights=weights)
    com_y = np.average(y, weights=weights)
    
    # Recentered coordinates
    x_prime = x - com_x
    y_prime = y - com_y
    
    # Inertia Tensor
    Ixx = np.sum(weights * y_prime**2)
    Iyy = np.sum(weights * x_prime**2)
    Ixy = -np.sum(weights * x_prime * y_prime)
    
    tensor = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    
    # Eigenvalues/vectors
    evals, evecs = np.linalg.eig(tensor)
    
    # Principal axis is the eigenvector corresponding to SMALLEST Moment of Inertia
    # (The axis the matter is distributed ALONG)
    # Wait, Inertia Tensor measures resistance to rotation.
    # Elongated object along X axis has LOW Ixx? No, Ixx = sum(y^2). 
    # If along X, y is small, so Ixx is small. Correct.
    
    min_idx = np.argmin(evals)
    principal_vector = evecs[:, min_idx]
    
    angle_rad = np.arctan2(principal_vector[1], principal_vector[0])
    return np.degrees(angle_rad)

def run_alignment_test(bias_rotation=False):
    print("="*60)
    print("COSMIC ALIGNMENT TEST (The 'Axis of Evil')")
    print("="*60)
    
    if bias_rotation:
        print("NOTE: Running with 'Rotating Universe' Bias (Simulating Pre-Existing Spin)")
    
    # 1. Generate Field
    # We need a large scale to see Quadrupole
    N = 128
    
    # If simulating "Bias", we stretch the grid slightly or modify spectral index directionally
    # But for now, let's run the standard isotropic generator and see if a "Fluke" happens
    # Or if the filaments naturally align.
    
    params = ScaffoldParameters(grid_size=N, box_size=500.0, smoothing_scale=5.0, random_seed=None)
    scaffold = DarkMatterScaffold(params)
    density = scaffold.generate()
    
    # If Bias, we explicitly squash the density along Z to simulate rotation
    if bias_rotation:
        # Simple compression along Z axis (Spin Axis)
        # This makes structures elongated in XY plane?
        # Rotation flattens poles. Statistics change.
        pass # The Moment calculation below works on the 2D Project (XY plane).
             # If we project along Z, we see the "TOP DOWN" view?
             # Let's project along arbitrary axis.
    
    # 2. Project to Sky (2D)
    sky_map = np.sum(density, axis=2)
    
    # 3. Filter for Scales (Multipoles)
    # Quadrupole (l=2) is Very Large Scale. Octupole (l=3) is slightly smaller.
    
    # Low-Pass Filter for Quadrupole (Main "Blob" orientation)
    quadrupole_map = gaussian_filter(sky_map, sigma=N/4)
    
    # Band-Pass Filter for Octupole (Smaller blobs)
    # Difference of Gaussians
    octupole_map = gaussian_filter(sky_map, sigma=N/8) - gaussian_filter(sky_map, sigma=N/4)
    
    # 4. Measure Axes
    angle_quad = calculate_principal_axis(quadrupole_map)
    angle_oct = calculate_principal_axis(octupole_map)
    
    # Normalize to -90 to 90
    angle_quad = angle_quad % 180
    if angle_quad > 90: angle_quad -= 180
    
    angle_oct = angle_oct % 180
    if angle_oct > 90: angle_oct -= 180
    
    alignment_diff = abs(angle_quad - angle_oct)
    if alignment_diff > 90: alignment_diff = 180 - alignment_diff
    
    print(f"1. Quadrupole Axis (l=2): {angle_quad:.1f} deg")
    print(f"2. Octupole Axis   (l=3): {angle_oct:.1f} deg")
    print(f"3. Misalignment:          {alignment_diff:.1f} deg")
    
    # 5. Verdict
    print("\n" + "="*60)
    print("VERDICT: AXIS OF EVIL")
    
    # In random fields, mean misalignment is 45 degrees.
    # "Aligned" means < 20 degrees.
    
    if alignment_diff < 20.0:
        print("RESULT: ALIGNED (+)")
        print("The Quadrupole and Octupole are anomalously aligned.")
        print("Matches 'Axis of Evil' observation!")
    else:
        print("RESULT: RANDOM (-)")
        print("Axes are uncorrelated. Standard isotropic result.")
    print("="*60)
    
    # 6. Visualize
    plt.figure(figsize=(10, 5), facecolor='black')
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(quadrupole_map, cmap='twilight', origin='lower')
    ax1.set_title(f"Quadrupole (l=2)\nAxis: {angle_quad:.1f}°", color='white')
    ax1.axis('off')
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(octupole_map, cmap='twilight', origin='lower')
    ax2.set_title(f"Octupole (l=3)\nAxis: {angle_oct:.1f}°", color='white')
    ax2.axis('off')
    
    plt.suptitle(f"Cosmic Alignment Test\nMisalignment: {alignment_diff:.1f}°", color='white', fontsize=14)
    plt.savefig('alignment_test_result.png', dpi=100, facecolor='black')
    print("   Saved plot to alignment_test_result.png")

if __name__ == "__main__":
    # Run loop to see if random seed ever produces alignment (the "Fluke" hypothesis)
    # vs if it happens often.
    run_alignment_test()
