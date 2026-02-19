"""
Dark Matter Scaffold Generator
==============================

Generates a pre-existing 3D dark matter web structure using Gaussian Random Fields.
This represents the "scaffolding" that existed before the Big Bang in the 
Dark Scaffold cosmological theory.

The cosmic web is modeled using:
1. A Gaussian Random Field with a power spectrum tuned to produce filamentary structure
2. Density thresholding to identify nodes, filaments, sheets, and voids
3. Configurable parameters for filament width and density contrast

Author: Rob Simens
Theory: Pre-Existing Dark Scaffold Cosmology
"""

import os
import gc
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn, fftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import Tuple, Optional
from corsair_io import enforce_corsair_root, safe_savefig


@dataclass
class ScaffoldParameters:
    """Configuration for the dark matter scaffold generation."""
    
    # Grid parameters
    grid_size: int = 128          # Number of cells per dimension
    box_size: float = 500.0       # Physical size in Mpc
    
    # Power spectrum parameters
    spectral_index: float = -1.5  # n_s for filamentary structure (more negative = more large-scale power)
    amplitude: float = 1.0        # Overall amplitude of fluctuations
    
    # Filament parameters
    smoothing_scale: float = 2.0  # Gaussian smoothing in grid cells
    filament_threshold: float = 0.5  # Density threshold for filament identification
    
    # Seed for reproducibility
    random_seed: Optional[int] = None
    
    @property
    def cell_size(self) -> float:
        """Physical size of each grid cell in Mpc."""
        return self.box_size / self.grid_size


class DarkMatterScaffold:
    """
    Generates and stores a 3D dark matter scaffold structure.
    
    This represents the primordial dark matter web that, according to the 
    Dark Scaffold theory, existed before the Big Bang. Baryonic matter
    would later "seep" into this structure.
    """
    
    def __init__(self, params: Optional[ScaffoldParameters] = None):
        """Initialize the scaffold generator with given parameters."""
        self.params = params or ScaffoldParameters()
        self.density_field = None
        self.filament_mask = None
        self._rng = np.random.default_rng(self.params.random_seed)
        
    def generate(self) -> np.ndarray:
        """
        Generate the dark matter scaffold using a Gaussian Random Field.
        
        Returns:
            3D numpy array representing the dark matter density field.
            Values are normalized such that mean=0, with positive values
            indicating overdensities (filaments, nodes) and negative values
            indicating underdensities (voids).
        """
        n = self.params.grid_size
        
        # Step 1: Generate white noise in Fourier space
        noise_real = self._rng.standard_normal((n, n, n))
        noise_imag = self._rng.standard_normal((n, n, n))
        noise_k = noise_real + 1j * noise_imag
        
        # Step 2: Create power spectrum P(k) ∝ k^n
        # This determines the scale distribution of structure
        kx = fftfreq(n, d=self.params.cell_size)
        ky = fftfreq(n, d=self.params.cell_size)
        kz = fftfreq(n, d=self.params.cell_size)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Avoid division by zero at k=0
        k_magnitude[0, 0, 0] = 1.0
        
        # Power spectrum: P(k) ∝ k^n_s
        # More negative spectral index = more large-scale power = more filamentary
        power_spectrum = self.params.amplitude * (k_magnitude ** self.params.spectral_index)
        power_spectrum[0, 0, 0] = 0  # Remove zero mode (sets mean to zero)
        
        # Step 3: Apply power spectrum to noise
        field_k = noise_k * np.sqrt(power_spectrum)
        
        # Step 4: Transform back to real space
        density_field = np.real(ifftn(field_k))
        
        # Step 5: Apply Gaussian smoothing to create filament structure
        density_field = gaussian_filter(
            density_field, 
            sigma=self.params.smoothing_scale
        )
        
        # Step 6: Normalize
        density_field = (density_field - np.mean(density_field)) / np.std(density_field)
        
        self.density_field = density_field
        self._identify_filaments()
        
        return self.density_field
    
    def _identify_filaments(self):
        """Identify filament regions based on density threshold."""
        if self.density_field is None:
            raise ValueError("Must generate density field first")
            
        # Filaments are regions above the threshold
        self.filament_mask = self.density_field > self.params.filament_threshold
        
    def get_filament_statistics(self) -> dict:
        """Calculate statistics about the scaffold structure."""
        if self.density_field is None:
            raise ValueError("Must generate density field first")
            
        # Volume fractions
        total_cells = self.density_field.size
        filament_cells = np.sum(self.filament_mask)
        
        # Mass fractions (assuming density ∝ (1 + δ))
        delta = self.density_field
        total_mass = np.sum(1 + delta)
        filament_mass = np.sum((1 + delta) * self.filament_mask)
        
        return {
            'volume_fraction_filaments': filament_cells / total_cells,
            'mass_fraction_filaments': filament_mass / total_mass,
            'mean_density': np.mean(self.density_field),
            'std_density': np.std(self.density_field),
            'max_overdensity': np.max(self.density_field),
            'min_underdensity': np.min(self.density_field),
            'grid_size': self.params.grid_size,
            'box_size_mpc': self.params.box_size,
        }
    
    def visualize_slice(self, axis: int = 2, slice_idx: Optional[int] = None, 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a 2D slice through the scaffold.
        
        Args:
            axis: Axis perpendicular to the slice (0=x, 1=y, 2=z)
            slice_idx: Index of the slice (default: middle)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.density_field is None:
            raise ValueError("Must generate density field first")
            
        if slice_idx is None:
            slice_idx = self.params.grid_size // 2
            
        # Extract slice
        if axis == 0:
            slice_data = self.density_field[slice_idx, :, :]
            xlabel, ylabel = 'Y (Mpc)', 'Z (Mpc)'
        elif axis == 1:
            slice_data = self.density_field[:, slice_idx, :]
            xlabel, ylabel = 'X (Mpc)', 'Z (Mpc)'
        else:
            slice_data = self.density_field[:, :, slice_idx]
            xlabel, ylabel = 'X (Mpc)', 'Y (Mpc)'
            
        # Create figure with dark theme for cosmic feel
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')
        
        # Physical extent
        extent = [0, self.params.box_size, 0, self.params.box_size]
        
        # Plot with cosmic color scheme
        im = ax.imshow(
            slice_data.T, 
            origin='lower',
            extent=extent,
            cmap='inferno',
            aspect='equal',
            vmin=-2, vmax=3
        )
        
        ax.set_xlabel(xlabel, color='white', fontsize=12)
        ax.set_ylabel(ylabel, color='white', fontsize=12)
        ax.set_title('Dark Matter Scaffold (Pre-Big Bang Structure)', 
                    color='white', fontsize=14)
        ax.tick_params(colors='white')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Density Contrast (δ)')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none')
            print(f"Saved visualization to {save_path}")
            plt.close(fig)
            
        return fig
    
    def visualize_3d(self, threshold: float = 1.0, 
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a 3D visualization of the scaffold structure.
        
        Args:
            threshold: Density threshold for plotting points
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.density_field is None:
            raise ValueError("Must generate density field first")
            
        # Find high-density regions
        z, y, x = np.where(self.density_field > threshold)
        densities = self.density_field[z, y, x]
        
        # Subsample if too many points
        max_points = 10000
        if len(x) > max_points:
            indices = self._rng.choice(len(x), max_points, replace=False)
            x, y, z = x[indices], y[indices], z[indices]
            densities = densities[indices]
        
        # Convert to physical coordinates
        cell_size = self.params.cell_size
        x_phys = x * cell_size
        y_phys = y * cell_size  
        z_phys = z * cell_size
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Scatter plot with color based on density
        scatter = ax.scatter(
            x_phys, y_phys, z_phys,
            c=densities,
            cmap='hot',
            s=1,
            alpha=0.6
        )
        
        ax.set_xlabel('X (Mpc)', color='white')
        ax.set_ylabel('Y (Mpc)', color='white')
        ax.set_zlabel('Z (Mpc)', color='white')
        ax.set_title('3D Dark Matter Scaffold\n(Pre-Existing Structure Before Big Bang)',
                    color='white', fontsize=12)
        
        # Style adjustments
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none')
            print(f"Saved 3D visualization to {save_path}")
            plt.close(fig)
            
        return fig


def main():
    """Demo the scaffold generator."""
    parser = argparse.ArgumentParser(description='Dark Matter Scaffold Generator')
    parser.add_argument('--hires', action='store_true',
                        help='Run at high resolution (256³ grid)')
    args = parser.parse_args()

    print("=" * 60)
    print("DARK MATTER SCAFFOLD GENERATOR")
    print("Pre-Existing Dark Scaffold Cosmology Theory")
    if args.hires:
        print("*** HIGH-RESOLUTION MODE ***")
    print("=" * 60)
    print()
    
    # ── Force all I/O to Corsair drive (disk8) ──────────────
    output_dir = enforce_corsair_root()

    # Create scaffold with specific parameters
    grid_size = 256 if args.hires else 128
    params = ScaffoldParameters(
        grid_size=grid_size,
        box_size=500.0,  # 500 Mpc box
        spectral_index=-1.5,  # Favors large-scale filamentary structure
        smoothing_scale=2.5,
        filament_threshold=0.5,
        random_seed=42  # For reproducibility
    )
    
    print("Generating dark matter scaffold...")
    print(f"  Grid: {params.grid_size}³ cells")
    print(f"  Box size: {params.box_size} Mpc")
    print(f"  Cell size: {params.cell_size:.2f} Mpc")
    print()
    
    scaffold = DarkMatterScaffold(params)
    density = scaffold.generate()
    
    # Print statistics
    stats = scaffold.get_filament_statistics()
    print("Scaffold Statistics:")
    print(f"  Volume in filaments: {stats['volume_fraction_filaments']*100:.1f}%")
    print(f"  Mass in filaments: {stats['mass_fraction_filaments']*100:.1f}%")
    print(f"  Max overdensity: δ = {stats['max_overdensity']:.2f}")
    print(f"  Min underdensity: δ = {stats['min_underdensity']:.2f}")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 2D slice
    fig_2d = scaffold.visualize_slice(
        axis=2, 
        save_path=os.path.join(output_dir, 'scaffold_slice.png')
    )
    plt.close('all')
    gc.collect()
    
    # 3D view
    fig_3d = scaffold.visualize_3d(
        threshold=1.0,
        save_path=os.path.join(output_dir, 'scaffold_3d.png')
    )
    plt.close('all')
    gc.collect()
    
    print()
    print("Dark matter scaffold generation complete!")
    print("This represents the pre-existing structure into which")
    print("baryonic matter will 'seep' after the Big Bang.")
    
    return scaffold


if __name__ == "__main__":
    scaffold = main()
