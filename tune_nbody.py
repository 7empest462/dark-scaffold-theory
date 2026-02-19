"""Quick parameter sweep for N-body fine-tuning."""
import sys
sys.path.insert(0, '/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory')

from nbody_simulation import FastNBodySimulation, FastNBodyParams
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters

print("=" * 60)
print("N-BODY FINE-TUNING")
print("=" * 60)
print()

# Generate scaffold
scaffold_params = ScaffoldParameters(
    grid_size=64, box_size=100.0, spectral_index=-1.5,
    smoothing_scale=2.0, filament_threshold=0.5, random_seed=42
)
scaffold = DarkMatterScaffold(scaffold_params)
scaffold.generate()
print("Scaffold ready")

# Test configurations: focus on very strong DM coupling
configs = [
    {"dm_coupling": 15.0, "self_gravity": 0.0, "damping": 0.995, "expansion_velocity": 50.0, "n_steps": 400},
    {"dm_coupling": 25.0, "self_gravity": 0.0, "damping": 0.998, "expansion_velocity": 70.0, "n_steps": 500},
    {"dm_coupling": 35.0, "self_gravity": 0.0, "damping": 0.999, "expansion_velocity": 80.0, "n_steps": 600},
]

best_corr = 0
best_sim = None

for i, cfg in enumerate(configs):
    print(f"\nTest {i+1}/3: dm={cfg['dm_coupling']}, exp={cfg['expansion_velocity']}")
    
    params = FastNBodyParams(
        n_particles=2000,
        dt=0.01,
        random_seed=42,
        **cfg
    )
    
    sim = FastNBodySimulation(scaffold, params)
    stats = sim.run(save_history=False)
    corr = stats['dm_baryon_correlation']
    
    print(f"  CORRELATION: {corr:.4f}")
    
    if corr > best_corr:
        best_corr = corr
        best_sim = sim
        best_cfg = cfg

print()
print("=" * 60)
print(f"BEST RESULT: {best_corr:.4f}")
print(f"Config: dm={best_cfg['dm_coupling']}, exp={best_cfg['expansion_velocity']}")
print("=" * 60)

best_sim.visualize_final('/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory/nbody_tuned.png')
print("\nSaved: nbody_tuned.png")
