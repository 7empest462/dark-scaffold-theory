"""
Fast N-Body Gravitational Simulation
=====================================

Optimized N-body dynamics using vectorized NumPy operations.
Uses direct summation with broadcasting for GPU-friendly computation.

For ~5000 particles, this is faster than Barnes-Hut in pure Python
due to NumPy's optimized array operations.

Physical model:
- Baryonic particles interact gravitationally with each other
- DM scaffold provides an external gravitational potential
- Particles cluster under self-gravity + scaffold attraction

Author: Rob Simens
Theory: Pre-Existing Dark Scaffold Cosmology
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fftn, ifftn, fftfreq
import time

from scaffold_generator import DarkMatterScaffold, ScaffoldParameters


# Physical constants (normalized units)
G_COSMO = 1.0


@dataclass
class FastNBodyParams:
    """Parameters for the fast N-body simulation."""
    
    # Particles
    n_particles: int = 5000
    
    # Time stepping
    dt: float = 0.005
    n_steps: int = 300
    
    # Physics
    softening: float = 1.0        # Softening length (Mpc)
    dm_coupling: float = 2.0      # Coupling to DM scaffold
    self_gravity: float = 0.3     # Self-gravity strength
    damping: float = 0.995        # Velocity damping
    
    # Initial conditions
    expansion_velocity: float = 20.0  # Initial radial velocity (Mpc/time)
    
    # Seed
    random_seed: Optional[int] = None
    

class FastNBodySimulation:
    """
    Fast N-body simulation using vectorized NumPy operations.
    
    Uses direct O(N²) summation but with broadcasting for speed.
    For typical simulation sizes (thousands of particles), this
    outperforms pure Python Barnes-Hut.
    """
    
    def __init__(self, scaffold: DarkMatterScaffold,
                 params: Optional[FastNBodyParams] = None):
        self.scaffold = scaffold
        self.params = params or FastNBodyParams()
        self.box_size = scaffold.params.box_size
        self._rng = np.random.default_rng(self.params.random_seed)
        
        # Particle state
        self.positions = None
        self.velocities = None
        self.masses = None
        
        # History
        self.position_history = []
        
        # Set up DM potential
        self._setup_dm_potential()
        
    def _setup_dm_potential(self):
        """Create interpolator for DM gravitational acceleration."""
        dm_field = self.scaffold.density_field
        n = self.scaffold.params.grid_size
        box = self.box_size
        
        # Solve Poisson equation in Fourier space: ∇²φ = 4πGρ
        rho_k = fftn(dm_field)
        
        kx = fftfreq(n, d=box/n) * 2 * np.pi
        ky = fftfreq(n, d=box/n) * 2 * np.pi
        kz = fftfreq(n, d=box/n) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_squared = KX**2 + KY**2 + KZ**2
        k_squared[0, 0, 0] = 1  # Avoid division by zero
        
        phi_k = -4 * np.pi * G_COSMO * rho_k / k_squared
        phi_k[0, 0, 0] = 0  # Remove DC component
        
        potential = np.real(ifftn(phi_k))
        potential = (potential - np.mean(potential)) / np.std(potential)
        
        # Calculate gradient (acceleration = -∇φ)
        grad_x = -sobel(potential, axis=0, mode='wrap') / (box/n)
        grad_y = -sobel(potential, axis=1, mode='wrap') / (box/n)
        grad_z = -sobel(potential, axis=2, mode='wrap') / (box/n)
        
        # Create interpolators
        x = np.linspace(0, box, n)
        self.dm_acc_interp = [
            RegularGridInterpolator((x, x, x), grad_x, bounds_error=False, fill_value=0),
            RegularGridInterpolator((x, x, x), grad_y, bounds_error=False, fill_value=0),
            RegularGridInterpolator((x, x, x), grad_z, bounds_error=False, fill_value=0),
        ]
        
    def get_dm_acceleration(self, positions: np.ndarray) -> np.ndarray:
        """Get acceleration from DM scaffold."""
        # Wrap positions to box
        pos_wrapped = np.mod(positions, self.box_size)
        
        acc = np.zeros_like(positions)
        for i in range(3):
            acc[:, i] = self.dm_acc_interp[i](pos_wrapped)
            
        return acc * self.params.dm_coupling
    
    def calculate_self_gravity(self, positions: np.ndarray, 
                               masses: np.ndarray) -> np.ndarray:
        """
        Calculate gravitational acceleration from particle self-gravity.
        Uses vectorized broadcasting for O(N²) but fast computation.
        """
        n = len(masses)
        
        # Displacement vectors: r_ij = r_j - r_i
        # Shape: [N, N, 3]
        r = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        
        # Handle periodic boundaries (minimum image convention)
        r = r - self.box_size * np.round(r / self.box_size)
        
        # Distance: |r_ij| with softening
        # Shape: [N, N]
        dist = np.sqrt(np.sum(r**2, axis=2) + self.params.softening**2)
        
        # Force: G * m_j * r_ij / |r_ij|³
        # Shape: [N, N, 3]
        force = G_COSMO * masses[np.newaxis, :, np.newaxis] * r / (dist[:, :, np.newaxis]**3)
        
        # Sum over j (exclude self-interaction handled by softening)
        # Shape: [N, 3]
        acceleration = np.sum(force, axis=1)
        
        return acceleration * self.params.self_gravity
    
    def initialize(self):
        """Initialize particles at Big Bang origin."""
        n = self.params.n_particles
        
        # Start at center with small spread
        origin = np.array([self.box_size/2, self.box_size/2, self.box_size/2])
        self.positions = origin + self._rng.normal(0, self.box_size * 0.01, (n, 3))
        
        # Radial explosion velocities
        directions = self.positions - origin
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)
        speeds = self._rng.exponential(self.params.expansion_velocity, n)
        self.velocities = directions * speeds[:, np.newaxis]
        
        # Add some random velocity for turbulence
        self.velocities += self._rng.normal(0, self.params.expansion_velocity * 0.1, (n, 3))
        
        # Equal masses
        self.masses = np.ones(n) / n
        
        print(f"Initialized {n} particles at Big Bang origin")
        print(f"  Mean velocity: {np.mean(np.linalg.norm(self.velocities, axis=1)):.2f} Mpc/unit")
    
    def step(self):
        """Advance simulation by one time step using velocity Verlet."""
        dt = self.params.dt
        
        # Get accelerations
        acc_self = self.calculate_self_gravity(self.positions, self.masses)
        acc_dm = self.get_dm_acceleration(self.positions)
        acc = acc_self + acc_dm
        
        # Update velocities (half step)
        self.velocities += 0.5 * acc * dt
        
        # Update positions
        self.positions += self.velocities * dt
        
        # Apply periodic boundary conditions
        self.positions = np.mod(self.positions, self.box_size)
        
        # Recalculate acceleration at new positions
        acc_self = self.calculate_self_gravity(self.positions, self.masses)
        acc_dm = self.get_dm_acceleration(self.positions)
        acc = acc_self + acc_dm
        
        # Update velocities (second half step)
        self.velocities += 0.5 * acc * dt
        
        # Apply damping (represents energy loss to radiation)
        self.velocities *= self.params.damping
    
    def run(self, save_history: bool = True) -> dict:
        """Run the full N-body simulation."""
        print(f"\nRunning fast N-body simulation...")
        print(f"  Particles: {self.params.n_particles}")
        print(f"  Steps: {self.params.n_steps}")
        print(f"  DM coupling: {self.params.dm_coupling}")
        print(f"  Self-gravity: {self.params.self_gravity}")
        print()
        
        self.initialize()
        start_time = time.time()
        
        if save_history:
            self.position_history = [self.positions.copy()]
        
        for t in range(self.params.n_steps):
            self.step()
            
            if save_history and (t + 1) % 30 == 0:
                self.position_history.append(self.positions.copy())
                
            if (t + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (t + 1) / elapsed
                eta = (self.params.n_steps - t - 1) / rate
                print(f"  Step {t + 1}/{self.params.n_steps} "
                      f"({rate:.1f} steps/s, ETA: {eta:.0f}s)")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation complete in {elapsed:.1f}s")
        
        return self.get_statistics()
    
    def get_statistics(self) -> dict:
        """Calculate statistics about final distribution."""
        density = self._calculate_density_field()
        dm_field = self.scaffold.density_field
        
        # Resize for comparison if needed
        if density.shape != dm_field.shape:
            from scipy.ndimage import zoom
            zoom_factor = dm_field.shape[0] / density.shape[0]
            density_resized = zoom(density, zoom_factor)
        else:
            density_resized = density
            
        correlation = np.corrcoef(dm_field.flatten(), density_resized.flatten())[0, 1]
        
        return {
            'dm_baryon_correlation': correlation,
            'velocity_dispersion': np.std(np.linalg.norm(self.velocities, axis=1)),
            'density_contrast': np.max(density) / (np.mean(density) + 1e-10),
            'position_std': np.std(self.positions),
        }
    
    def _calculate_density_field(self, n_grid: int = 64) -> np.ndarray:
        """Bin particles to density field."""
        density = np.zeros((n_grid, n_grid, n_grid))
        
        cell_indices = (self.positions / self.box_size * n_grid).astype(int)
        cell_indices = np.clip(cell_indices, 0, n_grid - 1)
        
        for i in range(len(self.positions)):
            ix, iy, iz = cell_indices[i]
            density[ix, iy, iz] += self.masses[i]
            
        return gaussian_filter(density, sigma=1)
    
    def visualize_final(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of final state."""
        fig = plt.figure(figsize=(16, 6), facecolor='black')
        
        # 3D scatter
        ax1 = fig.add_subplot(1, 3, 1, projection='3d', facecolor='black')
        
        n_show = min(3000, len(self.positions))
        indices = self._rng.choice(len(self.positions), n_show, replace=False)
        
        ax1.scatter(
            self.positions[indices, 0],
            self.positions[indices, 1],
            self.positions[indices, 2],
            c='cyan', s=1, alpha=0.6
        )
        
        ax1.set_xlabel('X (Mpc)', color='white')
        ax1.set_ylabel('Y (Mpc)', color='white')
        ax1.set_zlabel('Z (Mpc)', color='white')
        ax1.set_title('N-Body Final State', color='white', fontsize=14)
        ax1.tick_params(colors='white')
        ax1.set_xlim(0, self.box_size)
        ax1.set_ylim(0, self.box_size)
        ax1.set_zlim(0, self.box_size)
        
        # Density slice
        ax2 = fig.add_subplot(1, 3, 2, facecolor='#1a1a2e')
        density = self._calculate_density_field()
        ax2.imshow(density[:, :, density.shape[2]//2].T, origin='lower',
                  cmap='plasma', aspect='equal')
        ax2.set_title('Baryon Density (N-body)', color='white', fontsize=14)
        ax2.set_xlabel('X', color='white')
        ax2.set_ylabel('Y', color='white')
        ax2.tick_params(colors='white')
        
        # DM scaffold
        ax3 = fig.add_subplot(1, 3, 3, facecolor='#1a1a2e')
        dm_slice = self.scaffold.density_field[:, :, self.scaffold.params.grid_size//2]
        ax3.imshow(dm_slice.T, origin='lower', cmap='inferno', aspect='equal')
        ax3.set_title('DM Scaffold', color='white', fontsize=14)
        ax3.set_xlabel('X', color='white')
        ax3.set_ylabel('Y', color='white')
        ax3.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='black', bbox_inches='tight')
            print(f"Saved to {save_path}")
            
        return fig
    
    def create_evolution_animation(self, save_path: str):
        """Create animated GIF of the evolution."""
        if not self.position_history:
            print("No history saved - run with save_history=True")
            return
            
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        fig = plt.figure(figsize=(10, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        n_show = min(2000, len(self.position_history[0]))
        indices = self._rng.choice(len(self.position_history[0]), n_show, replace=False)
        
        def update(frame):
            ax.clear()
            positions = self.position_history[frame]
            ax.scatter(
                positions[indices, 0],
                positions[indices, 1],
                positions[indices, 2],
                c='cyan', s=1, alpha=0.6
            )
            ax.set_xlim(0, self.box_size)
            ax.set_ylim(0, self.box_size)
            ax.set_zlim(0, self.box_size)
            ax.set_title(f'N-Body Evolution (frame {frame+1}/{len(self.position_history)})',
                        color='white', fontsize=14)
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.set_zlabel('Z', color='white')
            ax.tick_params(colors='white')
            
        anim = FuncAnimation(fig, update, frames=len(self.position_history), interval=200)
        anim.save(save_path, writer=PillowWriter(fps=5), savefig_kwargs={'facecolor': 'black'})
        print(f"Saved animation to {save_path}")
        plt.close()


def main():
    """Run fast N-body simulation."""
    print("=" * 60)
    print("FAST N-BODY DARK SCAFFOLD SIMULATION")
    print("=" * 60)
    print()
    
    output_dir = '/Users/robsimens/Documents/Cosmology/dark-scaffold-theory'
    
    # Generate scaffold
    print("Generating dark matter scaffold...")
    scaffold_params = ScaffoldParameters(
        grid_size=64,
        box_size=100.0,
        spectral_index=-1.5,
        smoothing_scale=2.0,
        filament_threshold=0.5,
        random_seed=42
    )
    
    scaffold = DarkMatterScaffold(scaffold_params)
    scaffold.generate()
    print(f"  Generated {scaffold_params.grid_size}³ grid")
    
    # Run N-body simulation
    params = FastNBodyParams(
        n_particles=5000,
        n_steps=300,
        dt=0.005,
        dm_coupling=2.5,
        self_gravity=0.2,
        damping=0.997,
        expansion_velocity=15.0,
        random_seed=123
    )
    
    sim = FastNBodySimulation(scaffold, params)
    stats = sim.run(save_history=True)
    
    print()
    print("=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"  DM-Baryon Correlation: {stats['dm_baryon_correlation']:.4f}")
    print(f"  Velocity Dispersion: {stats['velocity_dispersion']:.4f}")
    print(f"  Density Contrast: {stats['density_contrast']:.1f}")
    print()
    
    # Save visualization
    sim.visualize_final(save_path=f'{output_dir}/nbody_result.png')
    
    # Create animation
    print("\nCreating evolution animation...")
    sim.create_evolution_animation(save_path=f'{output_dir}/nbody_evolution.gif')
    
    print()
    print("=" * 60)
    print("N-body simulation complete!")
    print("=" * 60)
    
    return sim


if __name__ == "__main__":
    sim = main()
