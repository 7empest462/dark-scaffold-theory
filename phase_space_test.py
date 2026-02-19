"""
Phase Space Analysis (Velocity vs Radius)
=========================================

Objective: Visualize the dynamical state of a massive halo/filament node.

Theory:
- Virialized Cluster: Shows a "trumpet" shape in phase space (infall streams + stable core).
- Uncollapsed Cloud: Shows only infall streams (no velocity dispersion in center).
- "Core-Cusp" Insight: If we see a stable core with low density slope (previous test),
  we have potentially solved the Core-Cusp problem naturally.

Method:
1. Run Seepage Simulation.
2. Find the most massive cluster center.
3. For all particles, plot Radial Velocity vs Radius (Phase Space).
4. Check for "Shell Crossing" and "Virialization".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from infall_simulation import SeepageSimulation, SeepageParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

def run_phase_space_test():
    print("="*60)
    print("PHASE SPACE ANALYSIS")
    print("Checking dynamical stability (Virialization)")
    print("="*60)
    
    # 1. Run Simulation
    print("\n1. Running Seepage Simulation...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=100.0, 
        random_seed=42
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=100000, 
        gravity_strength=80.0, # Stronger binding for phase space clarity
        friction=0.04,
        n_steps=600
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run()
    
    # 2. Find Center of Mass of Largest Clump
    print("\n2. Finding Cluster Center...")
    # Grid density
    n_grid = 64
    density = np.zeros((n_grid, n_grid, n_grid))
    idx = (sim.positions / sim.box * n_grid).astype(int)
    idx = np.clip(idx, 0, n_grid-1)
    np.add.at(density, (idx[:,0], idx[:,1], idx[:,2]), 1)
    
    # Find absolute peak
    peak_idx = np.unravel_index(np.argmax(density), density.shape)
    center = np.array(peak_idx) * (sim.box / n_grid)
    
    print(f"   Cluster Center found at {center}")
    
    # 3. Calculate Phase Space Coordinates (r, v_r)
    print("\n3. Calculating Phase Space (r, v_r)...")
    
    # Relative positions (Periodic BC handling)
    dx = sim.positions - center
    dx = dx - sim.box * np.round(dx / sim.box)
    radius = np.linalg.norm(dx, axis=1)
    
    # Select particles within 10 Mpc (Cluster environment)
    mask = radius < 15.0
    r_subset = radius[mask]
    dx_subset = dx[mask]
    v_subset = sim.velocities[mask]
    
    # Radial Velocity v_r = v . r_hat
    # r_hat = dx / radius
    # Avoid div/0
    r_safe = r_subset.copy()
    r_safe[r_safe == 0] = 1.0
    r_hat = dx_subset / r_safe[:, np.newaxis]
    
    v_radial = np.sum(v_subset * r_hat, axis=1)
    
    # 4. Plot
    print("\n4. Plotting Phase Space Diagram...")
    plt.figure(figsize=(10, 8), facecolor='black')
    
    # Scatter plot (s=1 for fine detail)
    plt.scatter(r_subset, v_radial, s=1, c='cyan', alpha=0.5, label='Baryons')
    
    # Add zero velocity line
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    
    # Typical Infall curve (Hubble flow + Infall)
    # v ~ -sqrt(GM/r)
    # Just for visual reference
    r_theory = np.linspace(0.1, 15, 100)
    v_infall = -10.0 / np.sqrt(r_theory) # Arbitrary scaling for 'freefall' shape
    plt.plot(r_theory, v_infall, color='orange', linestyle=':', label='Freefall Envelope (Approx)')
    
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Phase Space Diagram (Velocity vs Distance)", color='white', fontsize=14)
    ax.set_xlabel("Radius from Cluster Center [Mpc]", color='white')
    ax.set_ylabel("Radial Velocity (Infall < 0) [km/s approx]", color='white')
    ax.set_ylim(-15, 15)
    ax.tick_params(colors='white')
    plt.legend()
    
    plt.savefig('/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory/phase_space_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to phase_space_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: DYNAMICAL STATE")
    
    # Check dispersion in inner core (r < 2 Mpc)
    core_mask = r_subset < 2.0
    if np.sum(core_mask) > 10:
        v_disp = np.std(v_radial[core_mask])
        print(f"   Core Velocity Dispersion: {v_disp:.2f}")
        
        if v_disp > 2.0:
            print("RESULT: VIRIALIZED CORE ✅")
            print("Particles have randomized velocities (Hot Dynamic Core).")
        else:
            print("RESULT: COLD COLLAPSE ❄️")
            print("Particles are still streaming in (Cold Core).")
    else:
        print("RESULT: NO CORE FOUND")
        
    print("="*60)

if __name__ == "__main__":
    run_phase_space_test()
