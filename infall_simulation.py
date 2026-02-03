"""
Distributed Infall Simulation (The "Seepage" Model)
===================================================

Correction: Instead of trying to explode particles out of a central point vs strong gravity,
we model the state *immediately after* inflation/expansion.

1. START: Matter is distributed roughly uniformly (filled the volume).
2. DYNAMICS: Matter feels the pull of the pre-existing DM scaffold.
3. RESULT: Matter "seeps" (settles) into the filaments and nodes.

This matches the `cosmos.py` intuition: "The Fill (The Seepage)" filling "The Cracks".
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fftn, ifftn, fftfreq
import time
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

@dataclass 
class SeepageParams:
    n_particles: int = 5000
    n_steps: int = 1000
    dt: float = 0.005
    
    # Physics
    gravity_strength: float = 40.0 # Strong pull into wells
    friction: float = 0.02         # Cooling/damping to settle
    
    random_seed: Optional[int] = 42

class SeepageSimulation:
    def __init__(self, scaffold: DarkMatterScaffold, params: SeepageParams):
        self.scaffold = scaffold
        self.params = params
        self.box = scaffold.params.box_size
        self._rng = np.random.default_rng(params.random_seed)
        
        # Precompute potentials
        self._setup_potential()
        
    def _setup_potential(self):
        # 1. Get DM density
        rho = self.scaffold.density_field
        n = rho.shape[0]
        
        # 2. Amplify peaks (wells) for stronger capture
        rho_enhanced = np.clip(rho, 0, None) ** 1.2
        
        # 3. Solve Poisson for Potential: phi(k) = -rho(k)/k^2
        rhok = fftn(rho_enhanced)
        k = fftfreq(n, d=self.box/n) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        k2[0,0,0] = 1.0 # Avoid div/0
        
        phik = -rhok / k2
        phik[0,0,0] = 0
        phi = np.real(ifftn(phik))
        
        # Normalize potential
        self.phi = (phi - phi.mean()) / (np.std(phi) + 1e-10)
        
        # 4. Calculate Force Fields (-Gradient of Potential)
        # Gradient points UPHILL, so acceleration is -grad
        gx = -np.gradient(self.phi, self.box/n, axis=0)
        gy = -np.gradient(self.phi, self.box/n, axis=1)
        gz = -np.gradient(self.phi, self.box/n, axis=2)
        
        # Interpolators
        x = np.linspace(0, self.box, n)
        grid = (x, x, x)
        self.acc_x = RegularGridInterpolator(grid, gx, bounds_error=False, fill_value=0)
        self.acc_y = RegularGridInterpolator(grid, gy, bounds_error=False, fill_value=0)
        self.acc_z = RegularGridInterpolator(grid, gz, bounds_error=False, fill_value=0)

    def initialize(self):
        """Start with matter distributed EVERYWHERE (post-inflation)."""
        n = self.params.n_particles
        
        # Uniform distribution across the box
        self.positions = self._rng.uniform(0, self.box, (n, 3))
        
        # Small random initial velocities (thermal noise)
        self.velocities = self._rng.normal(0, 1.0, (n, 3))
        
        print(f"Initialized {n} particles distributed across volume.")

    def step(self):
        dt = self.params.dt
        
        # 1. Get Acceleration from DM Scaffold
        pos = np.mod(self.positions, self.box)
        acc = np.zeros_like(self.positions)
        acc[:,0] = self.acc_x(pos)
        acc[:,1] = self.acc_y(pos)
        acc[:,2] = self.acc_z(pos)
        
        # Scale gravity
        acc *= self.params.gravity_strength
        
        # 2. Update Velocity + Friction (Cooling)
        self.velocities += acc * dt
        self.velocities *= (1.0 - self.params.friction)
        
        # 3. Update Position
        self.positions += self.velocities * dt
        self.positions = np.mod(self.positions, self.box)

    def run(self):
        self.initialize()
        print("Simulating seepage...")
        
        start = time.time()
        for i in range(self.params.n_steps):
            self.step()
            if (i+1) % 200 == 0:
                print(f"  Step {i+1}/{self.params.n_steps}")
                
        print(f"Done in {time.time()-start:.2f}s")
        return self.analyze()

    def analyze(self):
        # 1. Bin particles
        n = self.scaffold.density_field.shape[0]
        density = np.zeros((n,n,n))
        idx = (self.positions / self.box * n).astype(int)
        idx = np.clip(idx, 0, n-1)
        # Vectorized binning
        np.add.at(density, (idx[:,0], idx[:,1], idx[:,2]), 1)
        
        # Smooth
        density = gaussian_filter(density, sigma=1.0)
        
        # 2. Correlate with DM
        dm = self.scaffold.density_field
        # Resize if needed (assuming 64^3 for now matches)
        
        corr = np.corrcoef(dm.flatten(), density.flatten())[0,1]
        
        return density, dm, corr

    def visualize(self, filename):
        density, dm, corr = self.analyze()
        n = density.shape[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
        
        # 3D Particles
        ax = axes[0]
        ax = fig.add_subplot(1,3,1, projection='3d', facecolor='black')
        # Subsample
        p = self.positions[::max(1, len(self.positions)//2000)] 
        ax.scatter(p[:,0], p[:,1], p[:,2], s=1, c='cyan', alpha=0.5)
        ax.set_title(f"Matter (Corr: {corr:.2f})", color='white')
        ax.set_xlim(0, self.box); ax.set_ylim(0, self.box); ax.set_zlim(0, self.box)
        ax.axis('off')

        # Baryon Density Slice
        mid = n // 2
        ax = axes[1]
        ax.imshow(density[:,:,mid], cmap='plasma', origin='lower')
        ax.set_title("Baryon Density", color='white')
        ax.axis('off')

        # DM Scaffold Slice
        ax = axes[2]
        ax.imshow(dm[:,:,mid], cmap='inferno', origin='lower')
        ax.set_title("DM Scaffold", color='white')
        ax.axis('off')

        plt.savefig(filename, facecolor='black', dpi=150)
        plt.close()
        print(f"Correlation: {corr:.4f}")
        print(f"Saved to {filename}")

if __name__ == '__main__':
    print("Generating Scaffold...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=100.0, spectral_index=-1.5,
        smoothing_scale=2.0, filament_threshold=0.5, random_seed=42
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=10000, 
        gravity_strength=80.0, # Very strong pull
        friction=0.05          # Significant cooling
    )
    
    sim = SeepageSimulation(scaffold, params)
    sim.run()
    sim.visualize('/Users/robsimens/Documents/Cosmology/dark-scaffold-theory/seepage_result.png')
