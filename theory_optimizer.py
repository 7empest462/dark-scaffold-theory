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

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
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
    best_score: float
    chi_squared: float
    n_evaluations: int
    optimization_time: float
    predictions: Dict[str, float]
    improvement: float


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
        Evaluate a parameter set and return negative score (for minimization).
        
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
            
            # Calculate score
            calc = TheoryLikelihoodCalculator(scaffold, sim.baryonic_density)
            score_data = calc.theory_score()
            score = score_data['total_score']
            
            # Track history
            self.history.append({
                'params': params_array.copy(),
                'score': score,
                'predictions': calc.predictions.copy()
            })
            
            if score > self.best_score_so_far:
                self.best_score_so_far = score
                if self.verbose and self.evaluation_count % 5 == 0:
                    print(f"  Eval {self.evaluation_count}: New best score = {score:.1f}")
                    
        except Exception as e:
            score = 0.0
            if self.verbose:
                print(f"  Eval {self.evaluation_count}: Error - {e}")
                
        return -score  # Negative for minimization
    
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
        
        # Calculate improvement
        initial_score = 25.0  # From our first run
        improvement = final_result['score'] - initial_score
        
        opt_result = OptimizationResult(
            best_params={
                'spectral_index': best_params[0],
                'smoothing_scale': best_params[1],
                'filament_threshold': best_params[2],
                'dm_attraction_strength': best_params[3],
                'filament_preference': best_params[4],
            },
            best_score=final_result['score'],
            chi_squared=final_result['chi_squared'],
            n_evaluations=self.evaluation_count,
            optimization_time=elapsed_time,
            predictions=final_result['predictions'],
            improvement=improvement
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
        score_data = calc.theory_score()
        
        return {
            'score': score_data['total_score'],
            'chi_squared': score_data['chi_squared'],
            'predictions': calc.predictions,
            'scaffold': scaffold,
            'simulation': sim,
            'calculator': calc
        }
    
    def _print_results(self, result: OptimizationResult):
        """Print optimization results."""
        print()
        print("=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print()
        print("Best Parameters Found:")
        for name, value in result.best_params.items():
            print(f"  {name}: {value:.4f}")
        print()
        print(f"Final Score: {result.best_score:.1f}/100")
        print(f"Improvement: {'+' if result.improvement > 0 else ''}{result.improvement:.1f} points")
        print(f"Chi-squared: {result.chi_squared:.2f}")
        print()
        print(f"Optimization took {result.optimization_time:.1f} seconds")
        print(f"Number of evaluations: {result.n_evaluations}")
        print()
        print("Predictions with optimized parameters:")
        for name, value in result.predictions.items():
            obs = OBSERVATIONAL_CONSTRAINTS.get(name)
            if obs:
                status = "✓" if abs(value - obs.observed_value) < 2*obs.uncertainty else "✗"
                print(f"  {status} {name}: {value:.4f} (observed: {obs.observed_value:.4f})")
            else:
                print(f"  - {name}: {value:.4f}")
        print()
        print("=" * 60)
    
    def visualize_optimization(self, result: OptimizationResult, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the optimization process and results."""
        fig = plt.figure(figsize=(16, 10), facecolor='black')
        
        # 1. Score improvement over iterations
        ax1 = fig.add_subplot(2, 2, 1, facecolor='#1a1a2e')
        
        if self.history:
            scores = [h['score'] for h in self.history]
            ax1.plot(scores, color='#3498db', linewidth=1, alpha=0.5)
            
            # Running maximum
            running_max = np.maximum.accumulate(scores)
            ax1.plot(running_max, color='#2ecc71', linewidth=2, label='Best so far')
            
            ax1.axhline(y=result.best_score, color='#e74c3c', linestyle='--', 
                       label=f'Final: {result.best_score:.1f}')
            ax1.axhline(y=25, color='white', linestyle=':', alpha=0.5,
                       label='Initial: 25.0')
            
        ax1.set_xlabel('Evaluation', color='white')
        ax1.set_ylabel('Theory Score', color='white')
        ax1.set_title('Optimization Progress', color='white')
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#1a1a2e', labelcolor='white')
        for spine in ax1.spines.values():
            spine.set_color('white')
            
        # 2. Parameter values comparison
        ax2 = fig.add_subplot(2, 2, 2, facecolor='#1a1a2e')
        
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
            
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(param_names)))
        bars = ax2.barh(range(len(param_names)), normalized, color=colors)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, param_values)):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', color='white', fontsize=10)
        
        ax2.set_yticks(range(len(param_names)))
        ax2.set_yticklabels([n.replace('_', '\n') for n in param_names], color='white', fontsize=9)
        ax2.set_xlabel('Normalized Parameter Value', color='white')
        ax2.set_title('Optimized Parameters', color='white')
        ax2.set_xlim(0, 1.3)
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('white')
            
        # 3. Before/After comparison (gauge)
        ax3 = fig.add_subplot(2, 2, 3, polar=True, facecolor='#1a1a2e')
        
        theta = np.linspace(0, np.pi, 100)
        colors_gauge = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
        for i, (start, end) in enumerate([(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]):
            ax3.fill_between(theta, 0.5, 1.0,
                            where=(theta >= start*np.pi) & (theta <= end*np.pi),
                            color=colors_gauge[i], alpha=0.7)
            
        # Before needle (25/100)
        before_angle = (1 - 25/100) * np.pi
        ax3.plot([before_angle, before_angle], [0, 0.8], color='gray', linewidth=2)
        ax3.scatter([before_angle], [0.8], c='gray', s=80, zorder=5, label='Before: 25')
        
        # After needle
        after_angle = (1 - result.best_score/100) * np.pi
        ax3.plot([after_angle, after_angle], [0, 0.9], color='white', linewidth=3)
        ax3.scatter([after_angle], [0.9], c='white', s=100, zorder=5, 
                   label=f'After: {result.best_score:.1f}')
        
        ax3.set_ylim(0, 1)
        ax3.set_theta_offset(np.pi)
        ax3.set_theta_direction(-1)
        ax3.set_thetamin(0)
        ax3.set_thetamax(180)
        ax3.set_rticks([])
        ax3.set_thetagrids([0, 45, 90, 135, 180], ['100', '75', '50', '25', '0'], color='white')
        ax3.set_title(f'Score Improvement: +{result.improvement:.1f}', color='white', pad=20)
        ax3.legend(loc='lower center', facecolor='#1a1a2e', labelcolor='white')
        
        # 4. Constraint satisfaction
        ax4 = fig.add_subplot(2, 2, 4, facecolor='#1a1a2e')
        
        constraint_names = []
        satisfactions = []
        
        for name, constraint in OBSERVATIONAL_CONSTRAINTS.items():
            if name in result.predictions:
                pred = result.predictions[name]
                obs = constraint.observed_value
                unc = constraint.uncertainty
                
                # Satisfaction: 1 = perfect, 0 = > 3 sigma away
                deviation = abs(pred - obs) / unc
                satisfaction = max(0, 1 - deviation / 3)
                
                constraint_names.append(name.replace('_', '\n'))
                satisfactions.append(satisfaction)
                
        colors_bar = ['#2ecc71' if s > 0.67 else '#f1c40f' if s > 0.33 else '#e74c3c' 
                     for s in satisfactions]
        
        bars = ax4.bar(range(len(constraint_names)), satisfactions, color=colors_bar)
        ax4.axhline(y=0.67, color='white', linestyle='--', alpha=0.3)
        ax4.set_xticks(range(len(constraint_names)))
        ax4.set_xticklabels(constraint_names, rotation=45, ha='right', fontsize=8, color='white')
        ax4.set_ylabel('Constraint Satisfaction', color='white')
        ax4.set_title('How Well Predictions Match Observations', color='white')
        ax4.set_ylim(0, 1.1)
        ax4.tick_params(colors='white')
        for spine in ax4.spines.values():
            spine.set_color('white')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='black')
            print(f"Saved optimization visualization to {save_path}")
            
        return fig


def main():
    """Run the optimization."""
    output_dir = '/Users/robsimens/Documents/Cosmology/dark-scaffold-theory'
    
    optimizer = TheoryOptimizer(n_quick_steps=50, verbose=True)
    result = optimizer.optimize(max_evaluations=80)
    
    # Save visualization
    optimizer.visualize_optimization(
        result,
        save_path=f'{output_dir}/optimization_results.png'
    )
    
    # Run full likelihood assessment with optimized parameters
    print()
    print("Running final assessment with optimized parameters...")
    print()
    
    scaffold_params = ScaffoldParameters(
        grid_size=100,
        box_size=500.0,
        spectral_index=result.best_params['spectral_index'],
        smoothing_scale=result.best_params['smoothing_scale'],
        filament_threshold=result.best_params['filament_threshold'],
        random_seed=42
    )
    
    scaffold = DarkMatterScaffold(scaffold_params)
    scaffold.generate()
    
    seep_params = SeepingParameters(
        n_particles=40000,
        n_timesteps=200,
        dm_attraction_strength=result.best_params['dm_attraction_strength'],
        filament_preference=result.best_params['filament_preference'],
        random_seed=123
    )
    
    sim = SeepingSimulation(scaffold, seep_params)
    sim.run(save_history=False)
    
    calc = TheoryLikelihoodCalculator(scaffold, sim.baryonic_density)
    report = calc.generate_report(
        save_path=f'{output_dir}/optimized_likelihood_report.txt'
    )
    print(report)
    
    calc.visualize_results(
        save_path=f'{output_dir}/optimized_likelihood_assessment.png'
    )
    
    # Save optimized parameters
    with open(f'{output_dir}/optimized_parameters.txt', 'w') as f:
        f.write("OPTIMIZED DARK SCAFFOLD PARAMETERS\n")
        f.write("=" * 40 + "\n\n")
        for name, value in result.best_params.items():
            f.write(f"{name}: {value}\n")
        f.write(f"\nFinal Score: {result.best_score}/100\n")
        f.write(f"Improvement: +{result.improvement} points\n")
        
    print(f"\nOptimized parameters saved to {output_dir}/optimized_parameters.txt")
    
    return result


if __name__ == "__main__":
    result = main()
