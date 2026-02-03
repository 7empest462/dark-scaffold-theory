"""
Energy Budget Calculator
========================

Calculates and compares the energy requirements for:
1. Standard Big Bang cosmology (creating all matter/energy)
2. Dark Scaffold theory (creating only baryonic matter into pre-existing DM)

This analysis helps determine if the Dark Scaffold model is energetically viable
and what energy scales would be required.

Physical constants and cosmological parameters are taken from Planck 2018 results.

Author: Rob Simens
Theory: Pre-Existing Dark Scaffold Cosmology
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt


# Fundamental Constants (SI units)
class PhysicsConstants:
    """Fundamental physics constants."""
    
    c = 2.998e8           # Speed of light (m/s)
    G = 6.674e-11         # Gravitational constant (m³/kg/s²)
    h = 6.626e-34         # Planck constant (J·s)
    hbar = 1.055e-34      # Reduced Planck constant (J·s)
    k_B = 1.381e-23       # Boltzmann constant (J/K)
    
    # Planck units
    t_P = np.sqrt(hbar * G / c**5)  # Planck time: ~5.39e-44 s
    l_P = np.sqrt(hbar * G / c**3)  # Planck length: ~1.62e-35 m
    m_P = np.sqrt(hbar * c / G)     # Planck mass: ~2.18e-8 kg
    E_P = m_P * c**2                # Planck energy: ~1.96e9 J
    T_P = m_P * c**2 / k_B          # Planck temperature: ~1.42e32 K


class CosmologicalParameters:
    """Cosmological parameters from Planck 2018."""
    
    # Hubble constant
    H_0 = 67.4  # km/s/Mpc
    H_0_SI = H_0 * 1000 / (3.086e22)  # Convert to s⁻¹
    
    # Density parameters (fraction of critical density)
    Omega_b = 0.0493      # Baryonic matter
    Omega_dm = 0.265      # Dark matter
    Omega_m = 0.315       # Total matter (Omega_b + Omega_dm)
    Omega_Lambda = 0.685  # Dark energy
    Omega_r = 9.0e-5      # Radiation (today, was dominant early on)
    
    # Critical density
    rho_crit = 3 * H_0_SI**2 / (8 * np.pi * PhysicsConstants.G)  # kg/m³
    
    # Observable universe
    r_universe = 4.4e26   # Radius in meters (~46.5 billion light years)
    V_universe = (4/3) * np.pi * r_universe**3  # Volume in m³
    
    # Age of universe
    t_universe = 13.8e9 * 3.154e7  # Age in seconds (~13.8 Gyr)
    
    # CMB temperature
    T_CMB = 2.725  # K (today)


@dataclass
class EnergyBudget:
    """Results of an energy budget calculation."""
    
    # Component energies (in Joules)
    E_baryonic: float
    E_dark_matter: float
    E_dark_energy: float
    E_radiation: float
    E_total: float
    
    # In more intuitive units
    E_total_ergs: float
    E_total_planck: float
    
    # Descriptive
    model_name: str
    
    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"\n{'='*60}",
            f"ENERGY BUDGET: {self.model_name}",
            f"{'='*60}",
            f"",
            f"Component Energies (Joules):",
            f"  Baryonic Matter:   {self.E_baryonic:.3e} J",
            f"  Dark Matter:       {self.E_dark_matter:.3e} J",
            f"  Dark Energy:       {self.E_dark_energy:.3e} J",
            f"  Radiation:         {self.E_radiation:.3e} J",
            f"  {'─'*40}",
            f"  TOTAL:             {self.E_total:.3e} J",
            f"",
            f"In Other Units:",
            f"  Total Energy:      {self.E_total_ergs:.3e} ergs",
            f"  Total Energy:      {self.E_total_planck:.3e} E_Planck",
            f"{'='*60}",
        ]
        return '\n'.join(lines)


class EnergyBudgetCalculator:
    """
    Calculates energy requirements for different cosmological models.
    
    Compares:
    1. Standard Big Bang: All matter/energy created at t=0
    2. Dark Scaffold: Only baryonic matter created; DM pre-exists
    3. Extended Dark Scaffold: Baryonic + radiation; DM + DE pre-exist
    """
    
    def __init__(self):
        self.constants = PhysicsConstants()
        self.cosmo = CosmologicalParameters()
        
    def calculate_mass_energy(self, density_fraction: float) -> float:
        """
        Calculate the mass-energy of a cosmic component.
        
        E = ρ * V * c²
        
        where ρ = Omega * ρ_crit
        
        Args:
            density_fraction: Omega parameter (fraction of critical density)
            
        Returns:
            Energy in Joules
        """
        rho = density_fraction * self.cosmo.rho_crit  # kg/m³
        mass = rho * self.cosmo.V_universe            # kg
        energy = mass * self.constants.c**2           # J
        return energy
    
    def standard_big_bang(self) -> EnergyBudget:
        """
        Calculate energy budget for standard Big Bang cosmology.
        
        All matter, dark matter, dark energy, and radiation are created.
        """
        E_b = self.calculate_mass_energy(self.cosmo.Omega_b)
        E_dm = self.calculate_mass_energy(self.cosmo.Omega_dm)
        E_de = self.calculate_mass_energy(self.cosmo.Omega_Lambda)
        E_r = self.calculate_mass_energy(self.cosmo.Omega_r)
        
        E_total = E_b + E_dm + E_de + E_r
        
        return EnergyBudget(
            E_baryonic=E_b,
            E_dark_matter=E_dm,
            E_dark_energy=E_de,
            E_radiation=E_r,
            E_total=E_total,
            E_total_ergs=E_total * 1e7,
            E_total_planck=E_total / self.constants.E_P,
            model_name="Standard Big Bang (ΛCDM)"
        )
    
    def dark_scaffold_minimal(self) -> EnergyBudget:
        """
        Calculate energy budget for Dark Scaffold theory (minimal).
        
        Only baryonic matter is created; dark matter pre-exists.
        Dark energy may or may not pre-exist (we assume it does here).
        Radiation is created from baryonic processes.
        """
        E_b = self.calculate_mass_energy(self.cosmo.Omega_b)
        E_r = self.calculate_mass_energy(self.cosmo.Omega_r)
        
        # DM and DE pre-exist, not created
        E_dm = 0
        E_de = 0
        
        E_total = E_b + E_r
        
        return EnergyBudget(
            E_baryonic=E_b,
            E_dark_matter=E_dm,
            E_dark_energy=E_de,
            E_radiation=E_r,
            E_total=E_total,
            E_total_ergs=E_total * 1e7,
            E_total_planck=E_total / self.constants.E_P,
            model_name="Dark Scaffold (DM + DE Pre-exist)"
        )
    
    def dark_scaffold_dm_only(self) -> EnergyBudget:
        """
        Calculate energy budget for Dark Scaffold theory (DM pre-exists only).
        
        Baryonic matter and dark energy are created.
        Only dark matter pre-exists.
        """
        E_b = self.calculate_mass_energy(self.cosmo.Omega_b)
        E_de = self.calculate_mass_energy(self.cosmo.Omega_Lambda)
        E_r = self.calculate_mass_energy(self.cosmo.Omega_r)
        
        # Only DM pre-exists
        E_dm = 0
        
        E_total = E_b + E_de + E_r
        
        return EnergyBudget(
            E_baryonic=E_b,
            E_dark_matter=E_dm,
            E_dark_energy=E_de,
            E_radiation=E_r,
            E_total=E_total,
            E_total_ergs=E_total * 1e7,
            E_total_planck=E_total / self.constants.E_P,
            model_name="Dark Scaffold (DM Pre-exists Only)"
        )
    
    def calculate_all(self) -> Tuple[EnergyBudget, EnergyBudget, EnergyBudget]:
        """Calculate all three energy budgets."""
        return (
            self.standard_big_bang(),
            self.dark_scaffold_dm_only(),
            self.dark_scaffold_minimal()
        )
    
    def energy_reduction_factor(self, 
                                 standard: EnergyBudget, 
                                 scaffold: EnergyBudget) -> float:
        """Calculate the energy reduction factor."""
        return standard.E_total / scaffold.E_total
    
    def visualize_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Create a visualization comparing energy budgets.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        standard, dm_only, minimal = self.calculate_all()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
        
        # Left: Bar chart of total energies
        ax1 = axes[0]
        ax1.set_facecolor('#1a1a2e')
        
        models = ['Standard\nBig Bang', 'Dark Scaffold\n(DM only)', 'Dark Scaffold\n(DM + DE)']
        energies = [standard.E_total, dm_only.E_total, minimal.E_total]
        
        # Normalize to standard Big Bang
        norm_energies = [e / standard.E_total * 100 for e in energies]
        
        colors = ['#e94560', '#fca311', '#00b4d8']
        bars = ax1.bar(models, norm_energies, color=colors, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, pct in zip(bars, norm_energies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{pct:.1f}%', ha='center', va='bottom', color='white', fontsize=12)
        
        ax1.set_ylabel('Energy Required (% of Standard Big Bang)', color='white', fontsize=12)
        ax1.set_title('Energy Budget Comparison', color='white', fontsize=14)
        ax1.tick_params(colors='white')
        ax1.set_ylim(0, 120)
        
        for spine in ax1.spines.values():
            spine.set_color('white')
        
        # Right: Stacked bar showing component breakdown
        ax2 = axes[1]
        ax2.set_facecolor('#1a1a2e')
        
        components = ['Baryonic', 'Dark Matter', 'Dark Energy', 'Radiation']
        
        standard_vals = [
            standard.E_baryonic / standard.E_total * 100,
            standard.E_dark_matter / standard.E_total * 100,
            standard.E_dark_energy / standard.E_total * 100,
            standard.E_radiation / standard.E_total * 100
        ]
        
        # For scaffold, show what's created vs pre-existing
        scaffold_created = [
            minimal.E_baryonic / standard.E_total * 100,
            0,  # DM pre-exists
            0,  # DE pre-exists  
            minimal.E_radiation / standard.E_total * 100
        ]
        
        x = np.arange(len(components))
        width = 0.35
        
        comp_colors = ['#00b4d8', '#9b59b6', '#e74c3c', '#f1c40f']
        
        bars1 = ax2.bar(x - width/2, standard_vals, width, label='Standard Big Bang',
                       color=comp_colors, edgecolor='white', linewidth=1)
        bars2 = ax2.bar(x + width/2, scaffold_created, width, label='Dark Scaffold',
                       color=comp_colors, edgecolor='white', linewidth=1, alpha=0.6,
                       hatch='///')
        
        ax2.set_xlabel('Energy Component', color='white', fontsize=12)
        ax2.set_ylabel('% of Standard Big Bang Total', color='white', fontsize=12)
        ax2.set_title('Component Breakdown', color='white', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(components, color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        
        for spine in ax2.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none')
            print(f"Saved energy comparison to {save_path}")
            
        return fig
    
    def calculate_phase_transition_energy(self) -> dict:
        """
        Calculate the energy released in a phase transition that could
        trigger the Big Bang in the Dark Scaffold model.
        
        In this model, the Big Bang might be caused by:
        1. False vacuum decay
        2. Collision of branes
        3. Phase transition in dark sector
        
        We estimate the energy scale required.
        """
        minimal = self.dark_scaffold_minimal()
        
        # The Big Bang needs to provide at least this much energy
        E_required = minimal.E_total
        
        # Temperature of the early universe at various epochs
        # T ~ E / k_B for characteristic energy
        T_required = E_required / (self.cosmo.V_universe * self.constants.k_B)
        
        # Compare to electroweak phase transition (~100 GeV ~ 10^15 K)
        T_EW = 1e15  # K
        
        # Compare to GUT scale (~10^16 GeV ~ 10^29 K)
        T_GUT = 1e29  # K
        
        return {
            'E_required_J': E_required,
            'E_per_planck_volume': E_required / self.cosmo.V_universe * self.constants.l_P**3,
            'comparison_to_EW': E_required / (T_EW * self.constants.k_B * self.cosmo.V_universe),
            'comparison_to_GUT': E_required / (T_GUT * self.constants.k_B * self.cosmo.V_universe),
            'energy_density': E_required / self.cosmo.V_universe,  # J/m³
        }


def main():
    """Run the energy budget analysis."""
    print("=" * 60)
    print("ENERGY BUDGET ANALYSIS")
    print("Dark Scaffold Cosmology Theory")
    print("=" * 60)
    
    calc = EnergyBudgetCalculator()
    
    # Calculate all budgets
    standard, dm_only, minimal = calc.calculate_all()
    
    # Print summaries
    print(standard.summary())
    print(dm_only.summary())
    print(minimal.summary())
    
    # Energy reduction factors
    print("\n" + "=" * 60)
    print("ENERGY REDUCTION ANALYSIS")
    print("=" * 60)
    
    reduction_dm_only = calc.energy_reduction_factor(standard, dm_only)
    reduction_minimal = calc.energy_reduction_factor(standard, minimal)
    
    print(f"\nDark Scaffold (DM pre-exists only):")
    print(f"  Energy reduction: {reduction_dm_only:.2f}x less energy required")
    print(f"  Savings: {(1 - 1/reduction_dm_only) * 100:.1f}%")
    
    print(f"\nDark Scaffold (DM + DE pre-exist):")
    print(f"  Energy reduction: {reduction_minimal:.2f}x less energy required")
    print(f"  Savings: {(1 - 1/reduction_minimal) * 100:.1f}%")
    
    # Phase transition analysis
    print("\n" + "=" * 60)
    print("PHASE TRANSITION REQUIREMENTS")
    print("=" * 60)
    
    phase = calc.calculate_phase_transition_energy()
    print(f"\nEnergy required for Big Bang in Dark Scaffold model:")
    print(f"  Total energy: {phase['E_required_J']:.3e} J")
    print(f"  Energy density: {phase['energy_density']:.3e} J/m³")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print()
    print("The Dark Scaffold theory requires significantly LESS energy")
    print("at the moment of the Big Bang, since dark matter and potentially")
    print("dark energy already exist as part of the pre-existing scaffold.")
    print()
    print("This makes the theory ENERGETICALLY FAVORABLE compared to")
    print("standard cosmology, where all matter/energy must be created ex nihilo.")
    
    # Create visualization
    output_dir = '/Users/robsimens/Documents/Cosmology/dark-scaffold-theory'
    calc.visualize_comparison(
        save_path=f'{output_dir}/energy_comparison.png'
    )
    
    print()
    print("=" * 60)
    
    return calc


if __name__ == "__main__":
    calc = main()
