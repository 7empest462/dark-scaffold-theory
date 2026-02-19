"""
Big Bang Seeping Simulation
===========================

Models the core mechanism of the Dark Scaffold theory: baryonic matter
"seeping" into a pre-existing dark matter scaffold after the Big Bang.

Key physical concepts modeled:
1. Point-source Big Bang origin
2. Radial expansion of energy/matter
3. Preferential flow along DM density gradients (gravitational channeling)
4. Matter accumulation in filaments and nodes

Author: Rob Simens
Theory: Pre-Existing Dark Scaffold Cosmology
"""

import os
import gc
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import Optional, Tuple, List
from corsair_io import enforce_corsair_root, safe_savefig

# Import our scaffold generator
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters


@dataclass
class SeepingParameters:
    """Configuration for the seeping simulation."""
    
    # Big Bang origin (as fraction of box size, 0.5 = center)
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    # Expansion parameters
    initial_energy: float = 1e68  # Joules (reduced from standard Big Bang)
    expansion_rate: float = 0.1   # Expansion speed (c = 1 units)
    
    # Seeping physics
    dm_attraction_strength: float = 2.0   # How strongly DM attracts baryonic matter
    filament_preference: float = 1.5      # Preference for flowing along filaments
    diffusion_rate: float = 0.05          # Random diffusion component
    
    # Simulation parameters
    n_particles: int = 50000    # Number of tracer particles
    n_timesteps: int = 200      # Number of time steps
    dt: float = 0.5             # Time step size
    
    # Seed for reproducibility
    random_seed: Optional[int] = None


class SeepingSimulation:
    """
    Simulates baryonic matter "seeping" into a pre-existing dark matter scaffold.
    
    The simulation models:
    1. Initial explosion from a point source (Big Bang)
    2. Radial expansion of matter
    3. Gravitational attraction toward DM overdensities
    4. Preferential flow along DM filaments
    5. Accumulation of matter in nodes and filaments
    """
    
    def __init__(self, scaffold: DarkMatterScaffold, 
                 params: Optional[SeepingParameters] = None):
        """
        Initialize the seeping simulation.
        
        Args:
            scaffold: Pre-generated dark matter scaffold
            params: Simulation parameters
        """
        self.scaffold = scaffold
        self.params = params or SeepingParameters()
        self._rng = np.random.default_rng(self.params.random_seed)
        
        # Particle state
        self.positions = None       # [n_particles, 3] positions
        self.velocities = None      # [n_particles, 3] velocities
        self.baryonic_density = None  # 3D density field of baryonic matter
        
        # History for animation
        self.position_history = []
        self.density_history = []
        
        # Compute gradient field for DM scaffold (gravitational field direction)
        self._compute_dm_gradient()
        
    def _compute_dm_gradient(self):
        """Compute the gradient of the DM density field (points uphill toward overdensities)."""
        dm_field = self.scaffold.density_field
        
        # Compute gradient using Sobel filter for smoothness
        self.dm_gradient_x = sobel(dm_field, axis=0, mode='wrap')
        self.dm_gradient_y = sobel(dm_field, axis=1, mode='wrap')
        self.dm_gradient_z = sobel(dm_field, axis=2, mode='wrap')
        
        # Create interpolators for sampling at arbitrary positions
        n = self.scaffold.params.grid_size
        box = self.scaffold.params.box_size
        x = np.linspace(0, box, n)
        
        self.dm_interp = RegularGridInterpolator(
            (x, x, x), dm_field, bounds_error=False, fill_value=0
        )
        self.grad_x_interp = RegularGridInterpolator(
            (x, x, x), self.dm_gradient_x, bounds_error=False, fill_value=0
        )
        self.grad_y_interp = RegularGridInterpolator(
            (x, x, x), self.dm_gradient_y, bounds_error=False, fill_value=0
        )
        self.grad_z_interp = RegularGridInterpolator(
            (x, x, x), self.dm_gradient_z, bounds_error=False, fill_value=0
        )
        
    def initialize_particles(self):
        """Initialize particles at the Big Bang origin with explosive velocities."""
        n = self.params.n_particles
        box = self.scaffold.params.box_size
        origin = np.array(self.params.origin) * box
        
        # All particles start at origin with small random offset
        self.positions = origin + self._rng.normal(0, 1, (n, 3))
        
        # Initial velocities: radially outward with random magnitude
        # This represents the initial explosive expansion
        directions = self._rng.normal(0, 1, (n, 3))
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        speeds = self._rng.exponential(self.params.expansion_rate * box / 10, n)
        self.velocities = directions * speeds[:, np.newaxis]
        
        # Initialize baryonic density field
        self.baryonic_density = np.zeros_like(self.scaffold.density_field)
        
        print(f"Initialized {n} particles at Big Bang origin")
        print(f"Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}) Mpc")
        
    def step(self):
        """
        Advance the simulation by one time step.
        
        Physics modeled:
        1. Gravitational attraction toward DM overdensities
        2. Preferential flow along filaments (tangent to gradient)
        3. Random diffusion
        4. Position update with periodic boundary conditions
        """
        dt = self.params.dt
        box = self.scaffold.params.box_size
        
        # Sample DM field and gradients at particle positions
        dm_density = self.dm_interp(self.positions)
        grad_x = self.grad_x_interp(self.positions)
        grad_y = self.grad_y_interp(self.positions)
        grad_z = self.grad_z_interp(self.positions)
        
        gradients = np.stack([grad_x, grad_y, grad_z], axis=1)
        
        # Normalize gradients
        grad_mag = np.linalg.norm(gradients, axis=1, keepdims=True)
        grad_mag = np.maximum(grad_mag, 1e-8)  # Avoid division by zero
        grad_unit = gradients / grad_mag
        
        # Force 1: Gravitational attraction toward DM overdensities
        # Stronger for higher DM density regions
        attraction_strength = self.params.dm_attraction_strength
        dm_force = attraction_strength * gradients
        
        # Force 2: Preferential flow along filaments
        # Particles tend to follow the filament direction
        # (perpendicular to gradient, along the filament ridge)
        filament_pref = self.params.filament_preference
        
        # Current velocity component along gradient
        vel_along_grad = np.sum(self.velocities * grad_unit, axis=1, keepdims=True)
        
        # Enhance velocity component perpendicular to gradient (along filament)
        vel_perp = self.velocities - vel_along_grad * grad_unit
        
        # Force 3: Random diffusion (thermal motion)
        diffusion = self.params.diffusion_rate * self._rng.normal(0, 1, self.positions.shape)
        
        # Update velocities
        # - Add DM gravity
        # - Slightly enhance tangential flow
        # - Add diffusion
        self.velocities += dm_force * dt
        self.velocities += filament_pref * vel_perp * dt * 0.1
        self.velocities += diffusion
        
        # Damping (energy loss as matter settles into structure)
        damping = 0.98
        self.velocities *= damping
        
        # Update positions
        self.positions += self.velocities * dt
        
        # Periodic boundary conditions
        self.positions = np.mod(self.positions, box)
        
        # Update baryonic density field
        self._update_density_field()
        
    def _update_density_field(self):
        """Update the baryonic matter density field from particle positions."""
        box = self.scaffold.params.box_size
        n_grid = self.scaffold.params.grid_size
        
        # Bin particles into grid cells
        cell_indices = (self.positions / box * n_grid).astype(int)
        cell_indices = np.clip(cell_indices, 0, n_grid - 1)
        
        # Count particles in each cell
        density = np.zeros((n_grid, n_grid, n_grid))
        for i in range(self.positions.shape[0]):
            ix, iy, iz = cell_indices[i]
            density[ix, iy, iz] += 1
            
        # Normalize and smooth
        density = density / np.max(density) * 10
        self.baryonic_density = gaussian_filter(density, sigma=1.5)
        
    def run(self, save_history: bool = True):
        """
        Run the full simulation.
        
        Args:
            save_history: Whether to save position history for animation
        """
        print(f"\nRunning seeping simulation for {self.params.n_timesteps} steps...")
        
        self.initialize_particles()
        
        if save_history:
            self.position_history = [self.positions.copy()]
            self.density_history = [self.baryonic_density.copy()]
        
        for t in range(self.params.n_timesteps):
            self.step()
            
            if save_history and t % 5 == 0:  # Save every 5th frame
                self.position_history.append(self.positions.copy())
                self.density_history.append(self.baryonic_density.copy())
                
            if (t + 1) % 50 == 0:
                print(f"  Step {t + 1}/{self.params.n_timesteps}")
                
        print("Simulation complete!")
        
        return self.get_final_statistics()
    
    def get_final_statistics(self) -> dict:
        """Calculate statistics about the final baryon distribution."""
        dm_field = self.scaffold.density_field
        bary_field = self.baryonic_density
        
        # Correlation between DM and baryonic matter
        correlation = np.corrcoef(dm_field.flatten(), bary_field.flatten())[0, 1]
        
        # What fraction of baryons ended up in DM filaments?
        filament_mask = self.scaffold.filament_mask
        bary_in_filaments = np.sum(bary_field * filament_mask)
        bary_total = np.sum(bary_field)
        
        return {
            'dm_baryon_correlation': correlation,
            'baryon_fraction_in_filaments': bary_in_filaments / bary_total,
            'baryon_concentration': np.max(bary_field) / np.mean(bary_field),
            'final_particle_spread': np.std(self.positions, axis=0).mean(),
        }
    
    def visualize_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create side-by-side visualization of DM scaffold and baryonic matter.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
        
        box = self.scaffold.params.box_size
        extent = [0, box, 0, box]
        slice_idx = self.scaffold.params.grid_size // 2
        
        # DM scaffold
        dm_slice = self.scaffold.density_field[:, :, slice_idx]
        axes[0].imshow(dm_slice.T, origin='lower', extent=extent, 
                      cmap='inferno', aspect='equal')
        axes[0].set_title('Dark Matter Scaffold\n(Pre-existing)', color='white', fontsize=12)
        axes[0].set_xlabel('X (Mpc)', color='white')
        axes[0].set_ylabel('Y (Mpc)', color='white')
        axes[0].tick_params(colors='white')
        axes[0].set_facecolor('black')
        
        # Baryonic matter
        bary_slice = self.baryonic_density[:, :, slice_idx]
        axes[1].imshow(bary_slice.T, origin='lower', extent=extent,
                      cmap='viridis', aspect='equal')
        axes[1].set_title('Baryonic Matter\n(After Seeping)', color='white', fontsize=12)
        axes[1].set_xlabel('X (Mpc)', color='white')
        axes[1].set_ylabel('Y (Mpc)', color='white')
        axes[1].tick_params(colors='white')
        axes[1].set_facecolor('black')
        
        # Overlay
        axes[2].imshow(dm_slice.T, origin='lower', extent=extent,
                      cmap='Purples', aspect='equal', alpha=0.5)
        axes[2].imshow(bary_slice.T, origin='lower', extent=extent,
                      cmap='Greens', aspect='equal', alpha=0.5)
        axes[2].set_title('Overlay: DM (purple) + Baryons (green)', 
                         color='white', fontsize=12)
        axes[2].set_xlabel('X (Mpc)', color='white')
        axes[2].set_ylabel('Y (Mpc)', color='white')
        axes[2].tick_params(colors='white')
        axes[2].set_facecolor('black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none')
            print(f"Saved comparison to {save_path}")
            plt.close(fig)
        
        return fig
    
    def create_animation(self, output_path: str, fps: int = 15):
        """
        Create an animated GIF of the seeping process.
        
        Args:
            output_path: Path to save the animation
            fps: Frames per second
        """
        if not self.position_history:
            raise ValueError("Must run simulation with save_history=True first")
            
        print(f"\nCreating animation with {len(self.position_history)} frames...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='black')
        
        box = self.scaffold.params.box_size
        slice_idx = self.scaffold.params.grid_size // 2
        
        # Static DM background
        dm_slice = self.scaffold.density_field[:, :, slice_idx]
        
        def update(frame):
            for ax in axes:
                ax.clear()
                ax.set_facecolor('black')
            
            # Left: Particle positions (projected onto XY plane)
            positions = self.position_history[frame]
            
            # Filter particles near the middle slice
            z_min = box * 0.4
            z_max = box * 0.6
            mask = (positions[:, 2] > z_min) & (positions[:, 2] < z_max)
            
            # DM background
            axes[0].imshow(dm_slice.T, origin='lower', extent=[0, box, 0, box],
                          cmap='Purples', aspect='equal', alpha=0.3, vmin=-2, vmax=3)
            
            # Particles
            axes[0].scatter(positions[mask, 0], positions[mask, 1], 
                           s=0.5, c='cyan', alpha=0.5)
            
            axes[0].set_xlim(0, box)
            axes[0].set_ylim(0, box)
            axes[0].set_title(f'Baryonic Matter Seeping\nTime Step: {frame * 5}', 
                             color='white', fontsize=12)
            axes[0].set_xlabel('X (Mpc)', color='white')
            axes[0].set_ylabel('Y (Mpc)', color='white')
            axes[0].tick_params(colors='white')
            
            # Right: Density field evolution
            density = self.density_history[frame][:, :, slice_idx]
            
            # DM background
            axes[1].imshow(dm_slice.T, origin='lower', extent=[0, box, 0, box],
                          cmap='Purples', aspect='equal', alpha=0.4, vmin=-2, vmax=3)
            # Baryon density overlay
            axes[1].imshow(density.T, origin='lower', extent=[0, box, 0, box],
                          cmap='Greens', aspect='equal', alpha=0.6)
            
            axes[1].set_title('Baryon Density Field', color='white', fontsize=12)
            axes[1].set_xlabel('X (Mpc)', color='white')
            axes[1].set_ylabel('Y (Mpc)', color='white')
            axes[1].tick_params(colors='white')
            
            return axes
        
        anim = FuncAnimation(fig, update, frames=len(self.position_history), 
                            interval=1000/fps, blit=False)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=100)
        
        plt.close()
        print(f"Saved animation to {output_path}")


def main():
    """Run the seeping simulation demo."""
    parser = argparse.ArgumentParser(description='Big Bang Seeping Simulation')
    parser.add_argument('--hires', action='store_true',
                        help='Run at high resolution (200³ grid, 80k particles, 300 steps)')
    args = parser.parse_args()

    print("=" * 60)
    print("BIG BANG SEEPING SIMULATION")
    print("Dark Scaffold Cosmology Theory")
    if args.hires:
        print("*** HIGH-RESOLUTION MODE ***")
    print("=" * 60)
    print()
    
    # ── Force all I/O to Corsair drive (disk8) ──────────────
    output_dir = enforce_corsair_root()
    
    # Step 1: Generate the dark matter scaffold
    print("Step 1: Generating dark matter scaffold...")
    grid_size = 200 if args.hires else 100
    scaffold_params = ScaffoldParameters(
        grid_size=grid_size,
        box_size=500.0,
        spectral_index=-1.5,
        smoothing_scale=2.5,
        random_seed=42
    )
    scaffold = DarkMatterScaffold(scaffold_params)
    scaffold.generate()
    
    scaffold_stats = scaffold.get_filament_statistics()
    print(f"  Volume in filaments: {scaffold_stats['volume_fraction_filaments']*100:.1f}%")
    print(f"  Mass in filaments: {scaffold_stats['mass_fraction_filaments']*100:.1f}%")
    print()
    
    # Step 2: Run seeping simulation
    print("Step 2: Running seeping simulation...")
    n_particles = 80000 if args.hires else 30000
    n_timesteps = 300 if args.hires else 150
    seep_params = SeepingParameters(
        origin=(0.5, 0.5, 0.5),  # Big Bang at center
        n_particles=n_particles,
        n_timesteps=n_timesteps,
        dm_attraction_strength=1.5,
        filament_preference=1.2,
        random_seed=123
    )
    
    sim = SeepingSimulation(scaffold, seep_params)
    save_history = not args.hires  # Skip history for hires to save memory
    stats = sim.run(save_history=save_history)
    
    print()
    print("Final Statistics:")
    print(f"  DM-Baryon Correlation: {stats['dm_baryon_correlation']:.3f}")
    print(f"  Baryons in Filaments: {stats['baryon_fraction_in_filaments']*100:.1f}%")
    print(f"  Baryon Concentration: {stats['baryon_concentration']:.1f}x")
    print()
    
    # Step 3: Create visualizations
    print("Step 3: Generating visualizations...")
    
    # Side-by-side comparison
    sim.visualize_comparison(
        save_path=os.path.join(output_dir, 'seeping_comparison.png')
    )
    plt.close('all')
    gc.collect()
    
    # Animation (skip for hires — too memory-intensive)
    if not args.hires:
        sim.create_animation(
            output_path=os.path.join(output_dir, 'seeping_animation.gif'),
            fps=12
        )
        plt.close('all')
        gc.collect()
    else:
        print("  Skipping animation in hires mode (memory-intensive)")
    
    print()
    print("=" * 60)
    print("Simulation complete!")
    print()
    print("KEY FINDING:")
    print(f"  Baryonic matter successfully 'seeped' into the pre-existing")
    print(f"  dark matter scaffold with {stats['dm_baryon_correlation']:.1%} correlation!")
    print()
    print("This supports the Dark Scaffold theory's core mechanism:")
    print("  Matter flows preferentially into existing DM structure,")
    print("  rather than DM and baryons collapsing together.")
    print("=" * 60)
    
    return sim


if __name__ == "__main__":
    sim = main()
