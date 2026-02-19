"""
Theory Parameter Optimizer
==========================

Uses optimization to find the scaffold parameters that best match
observational constraints while preserving the core theory dynamics.

This addresses the key tensions identified in the initial assessment:
1. Filling factor too high (38% vs 6% observed)
2. Correlation length mismatch
3. Power spectrum amplitude

Author: Rob Simens
Theory: Pre-Existing Dark Scaffold Cosmology
"""

import os
import gc
import argparse
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from corsair_io import enforce_corsair_root, safe_savefig
import time

from scaffold_generator import DarkMatterScaffold, ScaffoldParameters
from seeping_simulation import SeepingSimulation, SeepingParameters
from theory_likelihood import (
    TheoryLikelihoodCalculator, 
    OBSERVATIONAL_CONSTRAINTS,
    PowerSpectrumAnalyzer
)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, float]
    best_chi_squared: float
    best_reduced_chi_squared: float
    best_delta_bic: float
    n_evaluations: int
    optimization_time: float
    predictions: Dict[str, float]
    chi_squared_improvement: float


class TheoryOptimizer:
    """
    Optimizes Dark Scaffold theory parameters to maximize
    consistency with observations while preserving theoretical coherence.
    """
    
    def __init__(self, n_quick_steps: int = 50, verbose: bool = True):
        """
        Initialize the optimizer.
        
        Args:
            n_quick_steps: Number of seeping simulation steps for quick evaluation
            verbose: Print progress messages
        """
        self.n_quick_steps = n_quick_steps
        self.verbose = verbose
        self.evaluation_count = 0
        self.best_score_so_far = 0
        self.history = []
        
    def evaluate_parameters(self, params_array: np.ndarray) -> float:
        """
        Evaluate a parameter set and return chi-squared (for minimization).
        
        Parameter array structure:
        [0] spectral_index: -0.5 to -2.5
        [1] smoothing_scale: 1.0 to 5.0
        [2] filament_threshold: 0.5 to 2.5 (higher = fewer filaments)
        [3] dm_attraction_strength: 1.0 to 5.0
        [4] filament_preference: 0.5 to 3.0
        """
        self.evaluation_count += 1
        
        spectral_index = params_array[0]
        smoothing_scale = params_array[1]
        filament_threshold = params_array[2]
        dm_attraction = params_array[3]
        filament_pref = params_array[4]
        
        # Generate scaffold
        scaffold_params = ScaffoldParameters(
            grid_size=64,  # Smaller for speed
            box_size=500.0,
            spectral_index=spectral_index,
            smoothing_scale=smoothing_scale,
            filament_threshold=filament_threshold,
            random_seed=42
        )
        
        try:
            scaffold = DarkMatterScaffold(scaffold_params)
            scaffold.generate()
            
            # Quick seeping simulation
            seep_params = SeepingParameters(
                n_particles=10000,
                n_timesteps=self.n_quick_steps,
                dm_attraction_strength=dm_attraction,
                filament_preference=filament_pref,
                random_seed=123
            )
            
            sim = SeepingSimulation(scaffold, seep_params)
            sim.run(save_history=False)
            
            # Calculate chi-squared
            calc = TheoryLikelihoodCalculator(scaffold, sim.baryonic_density)
            stats = calc.statistical_summary()
            chi_sq = stats['total_chi_squared']
            
            # Track history
            self.history.append({
                'params': params_array.copy(),
                'chi_squared': chi_sq,
                'predictions': calc.predictions.copy()
            })
            
            # Track best (lowest) chi-squared
            if self.best_score_so_far == 0 or chi_sq < self.best_score_so_far:
                self.best_score_so_far = chi_sq
                if self.verbose and self.evaluation_count % 5 == 0:
                    print(f"  Eval {self.evaluation_count}: New best χ² = {chi_sq:.2f}")
                    
        except Exception as e:
            chi_sq = 1e10  # Massive penalty for failure
            if self.verbose:
                print(f"  Eval {self.evaluation_count}: Error - {e}")
                
        return chi_sq  # Positive for minimization
    
    def optimize(self, method: str = 'differential_evolution',
                 max_evaluations: int = 100) -> OptimizationResult:
        """
        Run optimization to find best parameters.
        
        Args:
            method: 'differential_evolution' or 'basinhopping'
            max_evaluations: Maximum number of evaluations
            
        Returns:
            OptimizationResult with best parameters and scores
        """
        if self.verbose:
            print("=" * 60)
            print("THEORY PARAMETER OPTIMIZATION")
            print("=" * 60)
            print()
        
        start_time = time.time()
        self.evaluation_count = 0
        self.best_score_so_far = 0
        self.history = []
        
        # Parameter bounds
        # [spectral_index, smoothing_scale, filament_threshold, dm_attraction, filament_pref]
        bounds = [
            (-2.5, -0.5),   # spectral_index
            (1.0, 5.0),     # smoothing_scale
            (1.0, 3.0),     # filament_threshold (key to fixing filling factor!)
            (1.0, 6.0),     # dm_attraction_strength
            (0.5, 3.0),     # filament_preference
        ]
        
        if self.verbose:
            print(f"Parameter bounds:")
            param_names = ['spectral_index', 'smoothing_scale', 'filament_threshold',
                          'dm_attraction', 'filament_preference']
            for name, (lo, hi) in zip(param_names, bounds):
                print(f"  {name}: [{lo}, {hi}]")
            print()
            print(f"Running optimization (max {max_evaluations} evaluations)...")
            print()
        
        # Run optimization
        result = differential_evolution(
            self.evaluate_parameters,
            bounds,
            maxiter=max_evaluations // 10,
            popsize=10,
            tol=0.01,
            seed=42,
            workers=1,  # Single-threaded for reproducibility
            disp=False
        )
        
        elapsed_time = time.time() - start_time
        
        # Get final evaluation with full simulation
        if self.verbose:
            print()
            print("Running full evaluation with best parameters...")
            
        best_params = result.x
        final_result = self._full_evaluation(best_params)
        
        # Calculate improvement (reduction in chi-squared)
        initial_chi_sq = 2500.0  # Approximate starting point
        improvement = initial_chi_sq - final_result['total_chi_squared']
        
        opt_result = OptimizationResult(
            best_params={
                'spectral_index': best_params[0],
                'smoothing_scale': best_params[1],
                'filament_threshold': best_params[2],
                'dm_attraction_strength': best_params[3],
                'filament_preference': best_params[4],
            },
            best_chi_squared=final_result['total_chi_squared'],
            best_reduced_chi_squared=final_result['reduced_chi_squared'],
            best_delta_bic=final_result['delta_bic'],
            n_evaluations=self.evaluation_count,
            optimization_time=elapsed_time,
            predictions=final_result['predictions'],
            chi_squared_improvement=improvement
        )
        
        if self.verbose:
            self._print_results(opt_result)
            
        return opt_result
    
    def _full_evaluation(self, params_array: np.ndarray) -> dict:
        """Run a full evaluation with more particles and time steps."""
        scaffold_params = ScaffoldParameters(
            grid_size=100,
            box_size=500.0,
            spectral_index=params_array[0],
            smoothing_scale=params_array[1],
            filament_threshold=params_array[2],
            random_seed=42
        )
        
        scaffold = DarkMatterScaffold(scaffold_params)
        scaffold.generate()
        
        seep_params = SeepingParameters(
            n_particles=30000,
            n_timesteps=150,
            dm_attraction_strength=params_array[3],
            filament_preference=params_array[4],
            random_seed=123
        )
        
        sim = SeepingSimulation(scaffold, seep_params)
        sim.run(save_history=False)
        
        calc = TheoryLikelihoodCalculator(scaffold, sim.baryonic_density)
        stats = calc.statistical_summary()
        
        return {
            'total_chi_squared': stats['total_chi_squared'],
            'reduced_chi_squared': stats['reduced_chi_squared'],
            'delta_bic': stats['delta_bic'],
            'predictions': calc.predictions,
            'scaffold': scaffold,
            'simulation': sim,
            'calculator': calc
        }
    
    def _print_results(self, result: OptimizationResult):
        """Print optimization results."""
        print()
        print("=" * 60)
        print("OPTIMIZATION RESULTS - STATISTICAL SUMMARY")
        print("=" * 60)
        print()
        print("Best Parameters Found:")
        for name, value in result.best_params.items():
            print(f"  {name}: {value:.4f}")
        print()
        print(f"Reduced χ²: {result.best_reduced_chi_squared:.2f}")
        print(f"Total χ²: {result.best_chi_squared:.2f}")
        print(f"Δ BIC vs ΛCDM: {result.best_delta_bic:.2f}")
        print(f"Improvement in χ²: {result.chi_squared_improvement:.2f}")
        print()
        print(f"Optimization took {result.optimization_time:.1f} seconds")
        print(f"Number of evaluations: {result.n_evaluations}")
        print()
        print("Predictions with optimized parameters:")
        for name, value in result.predictions.items():
            obs = OBSERVATIONAL_CONSTRAINTS.get(name)
            if obs:
                # Calculate sigma tension
                tension = (value - obs.observed_value) / obs.uncertainty
                abs_tension = abs(tension)
                status = "✓" if abs_tension < 1.0 else "~" if abs_tension < 3.0 else "!"
                print(f"  {status} {name}: {value:.4f} (Obs: {obs.observed_value:.4f}, Tension: {tension:+.1f}σ)")
            else:
                print(f"  - {name}: {value:.4f}")
        print()
        print("=" * 60)
    
    def visualize_optimization(self, result: OptimizationResult, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the optimization process and results."""
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = fig.add_gridspec(2, 2)
        
        # 1. Chi-squared improvement over iterations (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        if self.history:
            # Fliter out failed runs (massive chi-squared)
            scores = [h['chi_squared'] for h in self.history if h['chi_squared'] < 1e9]
            
            if scores:
                ax1.plot(scores, color='#3498db', linewidth=1, alpha=0.5, label='Evaluations')
                
                # Running minimum (best chi-squared so far)
                running_min = np.minimum.accumulate(scores)
                ax1.plot(running_min, color='#2ecc71', linewidth=2, label='Best so far')
                
                ax1.axhline(y=result.best_chi_squared, color='#e74c3c', linestyle='--', 
                           label=f'Final: {result.best_chi_squared:.2f}')
                
                # Log scale for y-axis often better for chi-squared
                ax1.set_yscale('log')
                
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Total χ² (Log Scale)')
        ax1.set_title('Optimization Progress', fontweight='bold')
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.2)
            
        # 2. Parameter values (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        param_names = list(result.best_params.keys())
        param_values = list(result.best_params.values())
        
        # Normalize parameters for visualization
        bounds_dict = {
            'spectral_index': (-2.5, -0.5),
            'smoothing_scale': (1.0, 5.0),
            'filament_threshold': (1.0, 3.0),
            'dm_attraction_strength': (1.0, 6.0),
            'filament_preference': (0.5, 3.0),
        }
        
        normalized = []
        for name, val in result.best_params.items():
            lo, hi = bounds_dict[name]
            normalized.append((val - lo) / (hi - lo))
            
        y_pos = np.arange(len(param_names))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(param_names)))
        
        ax2.barh(y_pos, normalized, color=colors, edgecolor='black')
        
        # Add value labels
        for i, (norm, val) in enumerate(zip(normalized, param_values)):
            ax2.text(norm + 0.02, i, f'{val:.2f}', va='center', fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([n.replace('_', '\n') for n in param_names])
        ax2.set_xlabel('Normalized Parameter Value (0=Min, 1=Max)')
        ax2.set_title('Optimized Parameters', fontweight='bold')
        ax2.set_xlim(0, 1.25)
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
            
        # 3. Statistical Summary Text (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        summary_text = (
            f"OPTIMIZATION RESULTS\n"
            f"====================\n\n"
            f"Best Reduced χ²: {result.best_reduced_chi_squared:.2f}\n"
            f"Best Total χ²: {result.best_chi_squared:.2f}\n"
            f"Δ BIC vs ΛCDM: {result.best_delta_bic:.2f}\n\n"
            f"Evaluations: {result.n_evaluations}\n"
            f"Time: {result.optimization_time:.1f}s\n"
        )
        
        ax3.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=14, family='monospace', 
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='black', alpha=0.9))
        
        # 4. Constraint Tensions (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        obs_names = []
        tensions = []
        
        for name, constraint in OBSERVATIONAL_CONSTRAINTS.items():
            if name in result.predictions:
                obs_names.append(name.replace('_', '\n'))
                pred = result.predictions[name]
                obs = constraint.observed_value
                unc = constraint.uncertainty
                tension = (pred - obs) / unc
                tensions.append(tension)
                
        y_pos = np.arange(len(obs_names))
        colors = ['#2ecc71' if abs(t) < 1 else '#f1c40f' if abs(t) < 3 else '#e74c3c' for t in tensions]
        
        ax4.barh(y_pos, tensions, color=colors, edgecolor='black')
        ax4.axvline(0, color='black', linewidth=1)
        ax4.axvspan(-1, 1, color='green', alpha=0.1, label='1σ')
        ax4.axvspan(-3, 3, color='yellow', alpha=0.1, label='3σ')
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(obs_names)
        ax4.set_xlabel('Tension (σ)')
        ax4.set_title('Final Constraint Tensions', fontweight='bold')
        ax4.legend()
        ax4.grid(axis='x', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved optimization visualization to {save_path}")
            plt.close(fig)
            
        return fig


def main():
    """Run the optimization."""
    parser = argparse.ArgumentParser(description='Dark Scaffold Theory Parameter Optimizer')
    parser.add_argument('--hires', action='store_true',
                        help='Run at high resolution (200³ grid, 100k particles, 400 steps, 150 evals)')
    args = parser.parse_args()

    if args.hires:
        print("*** HIGH-RESOLUTION MODE ***")

    # ── Force all I/O to Corsair drive (disk8) ──────────────
    output_dir = enforce_corsair_root()
    
    optimizer = TheoryOptimizer(n_quick_steps=50, verbose=True)
    max_evals = 150 if args.hires else 80
    result = optimizer.optimize(max_evaluations=max_evals)
    
    # Save visualization
    optimizer.visualize_optimization(
        result,
        save_path=os.path.join(output_dir, 'optimization_results.png')
    )
    
    # Run full likelihood assessment with optimized parameters
    print()
    print("Running final assessment with optimized parameters...")
    print()
    
    grid_size = 200 if args.hires else 100
    scaffold_params = ScaffoldParameters(
        grid_size=grid_size,
        box_size=500.0,
        spectral_index=result.best_params['spectral_index'],
        smoothing_scale=result.best_params['smoothing_scale'],
        filament_threshold=result.best_params['filament_threshold'],
        random_seed=42
    )
    
    scaffold = DarkMatterScaffold(scaffold_params)
    scaffold.generate()
    
    n_particles = 100000 if args.hires else 40000
    n_timesteps = 400 if args.hires else 200
    seep_params = SeepingParameters(
        n_particles=n_particles,
        n_timesteps=n_timesteps,
        dm_attraction_strength=result.best_params['dm_attraction_strength'],
        filament_preference=result.best_params['filament_preference'],
        random_seed=123
    )
    
    sim = SeepingSimulation(scaffold, seep_params)
    sim.run(save_history=False)
    
    calc = TheoryLikelihoodCalculator(scaffold, sim.baryonic_density)
    report = calc.generate_report(
        save_path=os.path.join(output_dir, 'optimized_likelihood_report.txt')
    )
    print(report)
    
    calc.visualize_results(
        save_path=os.path.join(output_dir, 'optimized_likelihood_assessment.png')
    )
    plt.close('all')
    gc.collect()
    
    # Save optimized parameters
    with open(os.path.join(output_dir, 'optimized_parameters.txt'), 'w') as f:
        f.write("OPTIMIZED DARK SCAFFOLD PARAMETERS\n")
        f.write("=" * 40 + "\n\n")
        for name, value in result.best_params.items():
            f.write(f"{name}: {value}\n")
        f.write(f"\nFinal Reduced Chi-Squared: {result.best_reduced_chi_squared:.2f}\n")
        f.write(f"Final Total Chi-Squared: {result.best_chi_squared:.2f}\n")
        f.write(f"Delta BIC: {result.best_delta_bic:.2f}\n")
        f.write(f"Chi-Squared Improvement: {result.chi_squared_improvement:.2f}\n")
        
    print(f"\nOptimized parameters saved to {output_dir}/optimized_parameters.txt")
    
    return result


if __name__ == "__main__":
    result = main()
