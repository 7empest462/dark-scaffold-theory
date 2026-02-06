"""
Entropy Efficiency Test (Thermodynamic Origin)
==============================================

Objective: Determine if the Dark Scaffold accelerates entropy production.
Hypothesis: The "Purpose" of the scaffold (thermodynamically speaking) is to 
act as a "Heat Sink" or "Entropy Maximizer" for the Big Bang.

If the Scaffold exists, baryonic matter should virialize (thermalize) FASTER 
than in a standard LambdaCDM cold start. This explains why structure and 
complexity emerged so quickly.

Method:
1. Simulate particle infall into:
   a) A Structured Scaffold (Our Model)
   b) A Random Gaussian Field (Standard/Random Model)
2. Track "Phase Space Entropy" (S) over time.
   S = - Integral( f(x,v) * ln(f(x,v)) )
   Approximation: S ~ ln(Volume of Phase Space occupied)
   Or simpler: Velocity dispersion increase rate (Thermalization).
3. Compare the rate dS/dt.

"""

import numpy as np
import matplotlib.pyplot as plt
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def calculate_approx_entropy(velocities, positions):
    """
    Calculates proxy for entropy based on velocity dispersion (Thermal energy).
    In a virializing system, potential energy turns into kinetic heat.
    Higher kinetic dispersion = Higher Entropy state for the gas.
    """
    # Better Proxy: Phase Space Volume
    # S ~ ln(Volume)
    # Volume ~ (sigma_x * sigma_v)
    
    sigma_x = np.std(positions, axis=0) # Spatial spread
    sigma_v = np.std(velocities, axis=0) # Velocity spread
    
    # Total phase space volume (product of spreads)
    # We sum the logs to get entropy
    entropy = np.sum(np.log(sigma_x + 1e-6) + np.log(sigma_v + 1e-6))
    return entropy

def simulate_infall_entropy(scaffold_density, n_particles=5000, steps=100):
    """
    Simulates test particles falling into the potential and tracks thermalization.
    """
    N = scaffold_density.shape[0]
    
    # 1. Initialize Particles (Cold Start)
    # Random positions, Zero velocity
    pos = np.random.rand(n_particles, 3) * N
    vel = np.zeros((n_particles, 3))
    
    # 2. Get Acceleration Field (Gradient of Potential)
    # Grad(Phi) ~ Grad(Density) approximation for local flows 
    # (Exact Poission is better but grad(rho) works for qualitative "pull")
    grad_x, grad_y, grad_z = np.gradient(scaffold_density)
    
    entropy_history = []
    
    dt = 0.5
    friction = 0.05 # Dynamical friction / gas drag
    
    for t in range(steps):
        # Interpolate force at particle positions
        # Nearest neighbor interpolation for speed
        ix = np.clip(pos[:,0].astype(int), 0, N-1)
        iy = np.clip(pos[:,1].astype(int), 0, N-1)
        iz = np.clip(pos[:,2].astype(int), 0, N-1)
        
        ax = grad_x[ix, iy, iz]
        ay = grad_y[ix, iy, iz]
        az = grad_z[ix, iy, iz]
        
        # Update Velocity (Acceleration + Drag)
        # Pull towards high density (gradient points TO peaks)
        vel += np.stack([ax, ay, az], axis=1) * dt
        vel *= (1.0 - friction) # Energy dissipation (Entropy generation via heating)
        
        # Update Position
        pos += vel * dt
        
        # Periodic Boundary
        pos = pos % N
        
        # Calculate Entropy Proxy (Phase Space Volume)
        S = calculate_approx_entropy(vel, pos)
        entropy_history.append(S)
        
    return np.array(entropy_history)

def run_entropy_test():
    print("="*60)
    print("ENTROPY EFFICIENCY TEST (Thermodynamic Purpose)")
    print("="*60)
    
    N = 64
    
    # 1. Structured Scaffold (The Theory)
    print("1. Simulating Infall into Dark Scaffold...")
    params_scaffold = ScaffoldParameters(grid_size=N, spectral_index=-1.5, random_seed=42)
    scaffold = DarkMatterScaffold(params_scaffold)
    rho_scaffold = scaffold.generate()
    entropy_scaffold = simulate_infall_entropy(rho_scaffold)
    
    # 2. Random Field (The Control / Standard Model approximation)
    print("2. Simulating Infall into Random Noise (No Pre-Structure)...")
    # White noise (spectral index 0) or just unsmoothed random
    params_random = ScaffoldParameters(grid_size=N, spectral_index=0.0, smoothing_scale=0.5, random_seed=999)
    random_field = DarkMatterScaffold(params_random)
    rho_random = random_field.generate()
    entropy_random = simulate_infall_entropy(rho_random)
    
    # 3. Analyze
    # Calculate Max Entropy and Rate
    max_S_scaffold = np.max(entropy_scaffold)
    max_S_random = np.max(entropy_random)
    
    ratio = max_S_scaffold / max_S_random
    
    print("\n3. Results:")
    print(f"   Max Entropy (Scaffold): {max_S_scaffold:.4f}")
    print(f"   Max Entropy (Random):   {max_S_random:.4f}")
    print(f"   Efficiency Ratio:       {ratio:.2f}x")
    
    # 4. Verdict
    print("\n" + "="*60)
    print("VERDICT: THERMODYNAMIC PURPOSE")
    
    if ratio > 1.5:
        print("RESULT: POSITIVE (+)")
        print(f"The Scaffold thermalizes matter {ratio:.1f}x faster/higher than random fields.")
        print("It acts as an efficient 'Entropy Engine'.")
    else:
        print("RESULT: NEGATIVE (-)")
        print("No significant thermodynamic advantage found.")
    print("="*60)
    
    # 5. Plot
    plt.figure(figsize=(10, 6), facecolor='black')
    plt.plot(entropy_scaffold, 'c-', linewidth=2, label='Dark Scaffold (Structured)')
    plt.plot(entropy_random, 'w--', alpha=0.5, label='Random Field (Unstructured)')
    
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Entropy Generation Rate", color='white', fontsize=14)
    ax.set_xlabel("Time Steps", color='white')
    ax.set_ylabel("System Temperature (Entropy Proxy)", color='white')
    ax.tick_params(colors='white')
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.savefig('entropy_test_result.png', dpi=100, facecolor='black')
    print("   Saved plot to entropy_test_result.png")

if __name__ == "__main__":
    run_entropy_test()
