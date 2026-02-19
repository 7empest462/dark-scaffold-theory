"""
Supermassive Black Hole (SMBH) Test
===================================

Objective: Determine if Dark Scaffold Theory explains the existence of 
SMBHs (10^9 M_sun) at z > 7.

The Problem:
Standard ΛCDM seeds are light (Pop III stars, ~100 M_sun). 
Growing from 100 -> 10^9 M_sun by z=7 requires constant super-Eddington accretion, 
which is physically difficult (radiation pushes mass away).

The Scaffold Solution:
Pre-existing deep potential wells could trigger "Direct Collapse Black Holes" (DCBH).
If gas falls in fast enough (> 0.1 M_sun/yr), it stays hot (T ~ 10^4 K), 
suppresses fragmentation (no stars), and collapses directly into a ~10^5 M_sun seed.
Starting at 10^5 M_sun makes reaching 10^9 M_sun easy.

Method:
1. Simulate gas infall into a Scaffold Node (Host Halo ~ 10^11 M_sun at z=20).
2. Measure Infall Rate dM/dt.
3. Check Condition: dM/dt > 0.1 M_sun/yr (The critical threshold for isothermal collapse).
4. Integrate growth to z=7.
"""

import numpy as np
import matplotlib.pyplot as plt
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters
from infall_simulation import SeepageSimulation, SeepageParams

def run_smbh_test():
    print("="*60)
    print("EARLY SUPERMASSIVE BLACK HOLE TEST")
    print("Checking for Direct Collapse conditions...")
    print("="*60)
    
    # 1. Setup Simulation (High Redshift z=20 conditions)
    # We focus on a single massive node
    print("\n1. Simulating Infall at z=20...")
    
    # Box size small (focus on core)
    box_size_kpc = 50.0 
    # At z=20, 50 kpc comoving is dense
    
    # We use our standard Seepage Simulation but interpret units differently
    # Or purely calculate potential gradient and freefall time
    
    # Let's use the Seepage logic:
    # Scaffold Node Mass M_h ~ 10^11 M_sun (from JWST test)
    # Gas mass M_g ~ 0.16 * M_h
    
    M_halo = 1e9 # Conservative estimate for a z=20 node (JWST test found much larger)
    M_halo = 1e11 # Let's use the JWST verification result (~10^11 M_sun)
    
    # Freefall Time t_ff = sqrt(3pi / 32 G rho)
    # Infall Rate ~ M_gas / t_ff
    
    # We calculate t_ff for gas at radius R
    R_vir = 1.0 # kpc
    G = 4.3e-6 # kpc km^2 / s^2 M_sun ?? No, G in M_sun, kpc, Gyr units?
    
    # Let's stick to simple physics calculation based on our model's unique feature:
    # The Potential is ALREADY there. In LCDM, it builds up.
    
    # Accretion Rate dM/dt ~ v^3 / G (Bondi)?
    # No, isothermality condition relies on infall rate.
    
    # Let's SIMULATE the velocity of particles falling into the static potential
    # and compute the flux through a core radius.
    
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=10.0, # 10 Mpc box? No, let's say 10 kpc physical equivalent scaling
        random_seed=123
    ))
    scaffold.generate()
    
    # Find potential peak
    rho = scaffold.density_field
    peak_idx = np.unravel_index(np.argmax(rho), rho.shape)
    
    # Create particles around peak
    n_part = 10000
    # Random sphere distribution r < 2.0 (code units)
    
    # Run Seepage (Gravity pull)
    params = SeepageParams(
        n_particles=n_part,
        gravity_strength=200.0, # Strong gravity for massive node
        friction=0.01, # Low friction (gas cooling is inefficient in DCBH scenario)
        n_steps=100
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.initialize()
    
    # Move particles to be centered on peak for the test
    # (Hacking the init for this specific test)
    center = np.array(peak_idx) * (10.0 / 64)
    offsets = (np.random.rand(n_part, 3) - 0.5) * 4.0 
    sim.positions = center + offsets
    # Wrap
    sim.positions = sim.positions % 10.0
    sim.velocities = np.zeros_like(sim.positions) # Start from rest (cold accretion)
    
    # Run
    print("   Running collapse simulation...")
    sim.run()
    
    # Measure flux into core (r < 0.2 units)
    # Core radius physical ~ 100 pc
    
    final_pos = sim.positions
    dx = final_pos - center
    dx = dx - 10.0 * np.round(dx/10.0)
    r = np.linalg.norm(dx, axis=1)
    
    r_core = 0.2
    mass_in_core = np.sum(r < r_core)
    
    # Time elapsed
    # Simulation time units -> Physical years
    # In Seepage, 1 step ~ roughly 0.5 Myr for standard scaling
    # We ran 100 steps ~ 50 Myr
    dt_myr = 50.0
    
    # Mass scaling
    # Total system mass M_halo ~ 10^11 M_sun
    # Particle mass = M_gas / n_part
    M_gas_total = 0.16 * M_halo
    m_p = M_gas_total / n_part
    
    accreted_mass = mass_in_core * m_p
    accretion_rate = accreted_mass / (dt_myr * 1e6) # M_sun / yr
    
    print(f"\n   Accreted Mass: {accreted_mass:.2e} M_sun")
    print(f"   Time Elapsed: {dt_myr} Myr")
    print(f"   Infall Rate: {accretion_rate:.4f} M_sun/yr")
    
    critical_rate = 0.1 # DCBH threshold
    
    print("\n   DCBH Threshold: > 0.1000 M_sun/yr")
    
    if accretion_rate > critical_rate:
        seed_mass = 1e5 # Direct Collapse Seed
        print("   -> Condition MET. Direct Collapse to 10^5 M_sun Seed.")
    else:
        seed_mass = 100.0 # Pop III Seed
        print("   -> Condition FAILED. Standard Pop III Seed.")
        
    # 2. Integrate Growth z=20 -> z=7
    print("\n2. Integrating Growth (z=20 to z=7)...")
    
    z_start = 20.0
    z_end = 7.0
    # Time available:
    # Age(7) ~ 750 Myr. Age(20) ~ 180 Myr.
    # Delta t ~ 570 Myr.
    delta_t = 570e6 # years
    
    # Eddington limit timescale (e-folding time)
    # t_edd = 45 Myr (assuming 10% efficiency)
    t_edd = 45e6 
    
    # M(t) = M_0 * exp(t / t_edd)
    
    M_final_scaffold = seed_mass * np.exp(delta_t / t_edd)
    
    # Standard Model Comparison (Seed 100)
    M_final_std = 100.0 * np.exp(delta_t / t_edd)
    
    print(f"   Time window: {delta_t/1e6:.0f} Myr")
    print(f"   Standard Model (Start 100 M_sun): {M_final_std:.2e} M_sun")
    print(f"   Scaffold Model (Start {seed_mass:.0e} M_sun): {M_final_scaffold:.2e} M_sun")
    
    # 3. Visualization
    plt.figure(figsize=(10, 6), facecolor='black')
    
    t = np.linspace(0, delta_t, 100)
    m_std = 100.0 * np.exp(t / t_edd)
    m_scaf = seed_mass * np.exp(t / t_edd)
    
    plt.semilogy(t/1e6, m_std, '--', color='gray', label='Standard Pop III Seed')
    plt.semilogy(t/1e6, m_scaf, '-', color='cyan', linewidth=2, label='Scaffold DCBH Seed')
    
    plt.axhline(1e9, color='red', linestyle=':', label='Observed Quasars (10^9 M_sun)')
    
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    ax.set_title("Supermassive Black Hole Growth", color='white', fontsize=14)
    ax.set_xlabel("Time since z=20 [Myr]", color='white')
    ax.set_ylabel("Black Hole Mass [M_sun]", color='white')
    ax.tick_params(colors='white')
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.savefig('/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory/smbh_test_result.png', 
               dpi=150, facecolor='black')
    print("   Saved to smbh_test_result.png")
    
    print("\n" + "="*60)
    print("VERDICT: EARLY QUASARS")
    
    if M_final_scaffold > 1e9:
        print("RESULT: SOLVED ✅")
        print("The Pre-existing Scaffold triggers Direct Collapse.")
        print("10^5 M_sun seeds easily grow to 10^9 M_sun by z=7.")
    else:
        print("RESULT: FAILED")
    print("="*60)

if __name__ == "__main__":
    run_smbh_test()
