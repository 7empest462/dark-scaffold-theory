"""
The Multiverse Tournament (Cosmological Natural Selection)
==========================================================

Objective: Test Lee Smolin's hypothesis that the laws of physics are 
"fine-tuned" because they are the result of evolutionary selection.

Hypothesis: Universes that produce more Black Holes (offspring) are selected for.
Our simulations should show that starting from random physics, the population 
converges to a "Sweet Spot" of parameters (matches 'Standard Model').

Method:
1.  **Define a Universe**:
    - Parameters: G (Gravity), Lambda (Dark Energy), Sigma_8 (Scaffold Density).
2.  **Simulation Step (The Life of a Universe)**:
    - Simulate structure formation using the approximated Press-Schechter formalism.
    - Calculate `N_BH` (Number of High-Mass halos that collapse into Black Holes).
    - Constraint: 
        - If Lambda is too high: Structure never forms (Heat Death) -> N_BH = 0.
        - If G is too high: Universe recollapses instantly (Big Crunch) -> N_BH = 1 (or low).
        - Sweet Spot: Long life + high clustering = Maximum BHs.
3.  **Genetic Algorithm**:
    - Population: 100 Universes.
    - Generations: 50.
    - Selection: Top 20% reproduce.
    - Mutation: Small random variations in G, Lambda.
4.  **Result**: Does the "Winning" G and Lambda match our reality?

"""

import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
from scipy.special import erfc

# --- PHYSICS CONSTANTS (Normalized to "Our Universe" = 1.0) ---
# We define our universe as G=1.0, Lambda=0.7 (approx), Sigma=0.8
OUR_G = 1.0
OUR_LAMBDA = 0.7 
OUR_SIGMA = 0.8

class Universe:
    def __init__(self, G, Lambda, Sigma):
        self.G = G
        self.Lambda = Lambda
        self.Sigma = Sigma # Initial density contrast
        self.fitness = 0.0
        self.lifespan = 0.0
        
    def simulate_life(self):
        """
        Simulate the cosmic history and count Black Holes.
        Physics Model: simplified Press-Schechter / Structure Growth.
        """
        # 1. Check for immediate invalid parameters
        if self.G <= 0: return 0.0 # No gravity? No universe.
        
        # 2. Calculate Growth Factor D(t)
        # In linear theory, Growth Rate f ~ Omega_m^0.55
        # Here we approximate: Growth depends on G (pull) vs Lambda (push).
        
        # Effective Clustering Rate 'R'
        # Higher G -> Faster clustering
        # Higher Lambda -> Expansion wins, clustering stops (Freeze out)
        
        # Freeze-out time (when Lambda dominates): T_freeze ~ 1 / sqrt(Lambda)
        if self.Lambda <= 0.01: 
            T_freeze = 100.0 # Effectively infinite (Big Crunch risk instead)
        else:
            T_freeze = 1.0 / np.sqrt(self.Lambda)
            
        # Crunch time (if G is too high and Lambda low)
        # T_crunch ~ 1 / sqrt(G)
        # If density is high enough. Let's assume standard density.
        
        # Let's model BH formation as an integration over time
        dt = 0.1
        current_time = 0.0
        n_black_holes = 0.0
        
        # Evolve
        while current_time < 20.0: # Max simulation time (e.g., 20 Gyr)
            current_time += dt
            
            # Expansion Factor 'a'
            # Simple Friedmann: H^2 ~ rho - k + Lambda/3
            # We simplify: Cosmic Scale 'a' grows.
            # If Lambda dominates, a grows exponentially -> Density drops -> Formation stops.
            
            # Critical Threshold for collapse delta_c ~ 1.686 / D(t)
            # If G is high, D(t) grows fast.
            
            # Rate of BH formation at this time step:
            # Proportional to likelihood of > 30 Sigma peaks (rare events)
            
            # Effective 'Sigma(t)' (Fluctuation amplitude today)
            # Growth slows down as Lambda kicks in.
            
            # Heuristic Model:
            # 1. Matter Density rho_m ~ 1/a^3
            # 2. Dark Energy rho_de = Lambda
            # 3. Collapse Power P_coll = G * rho_m
            # 4. Expansion Drag P_exp = Lambda
            
            # If P_exp > P_coll * Factor, structure formation freezes.
            
            non_linear_growth = (self.G * (current_time ** 0.6)) # Matter epoch growth
            
            # Dark Energy damping
            if self.Lambda > 0:
                damping = np.exp(-self.Lambda * current_time) 
            else:
                damping = 1.0 # If Lambda <=0, checking for crunch
                
            effective_sigma = self.Sigma * non_linear_growth * damping
            
            # Check for Big Crunch constraint (G too high, Lambda too low)
            if self.Lambda < 0.1 and self.G > 2.0 and current_time > 5.0:
                # Recollapse! Everything merges. 
                # Single Singularity. Not useful for reproduction?
                # Smolin argues universes must bounce. A Crunch is a bounce.
                # But a Crunch destroys the "Many Black Holes" scenario? 
                # Let's say we maximize *Discrete* Black Holes (N > 1).
                # A Crunch is N=1.
                self.lifespan = current_time
                return 1.0 # Only 1 offspring
                
            # BH Formation Rate
            # Using Press-Schechter tail: exp(-delta_c^2 / 2sigma^2)
            # Threshold delta_c = 1.686
            
            if effective_sigma > 0:
                prob_formation = np.exp(-1.686**2 / (2 * effective_sigma**2))
            else:
                prob_formation = 0
                
            # Volume helps! Expanding universe has more volume.
            # Vol ~ a^3 ~ time^2 approx (in matter domination)
            # But if Lambda is high, Volume is huge but Prob is 0.
            volume = current_time ** 1.5 
            
            new_bhs = prob_formation * volume * 100 # Scale factor
            n_black_holes += new_bhs
            
            # Cutoff if probability drops too low (Freeze out)
            if prob_formation < 1e-6:
                break
                
        self.lifespan = current_time
        self.fitness = n_black_holes
        return self.fitness

def run_tournament():
    print("="*60)
    print("THE MULTIVERSE TOURNAMENT (Simulating Cosmic Selection)")
    print("="*60)
    
    # 1. Initialize Population (Random Physics)
    pop_size = 100
    generations = 50
    population = []
    
    print(f"Initializing {pop_size} universes with random G and Lambda...")
    for _ in range(pop_size):
        # Random G between 0.0 and 5.0 (Our G = 1.0)
        # Random Lambda between 0.0 and 5.0 (Our Lambda = 0.7)
        g = np.random.uniform(0.1, 5.0)
        l = np.random.uniform(0.0, 5.0)
        s = np.random.uniform(0.1, 2.0) # Scaffold Density
        population.append(Universe(G=g, Lambda=l, Sigma=s))
        
    # Stats history
    avg_g_history = []
    avg_l_history = []
    best_fitness_history = []
    
    # 2. Evolution Loop
    for gen in range(generations):
        # A. Evaluate Fitness
        scores = []
        for univ in population:
            univ.simulate_life()
            scores.append(univ.fitness)
            
        scores = np.array(scores)
        
        # Stats
        mean_g = np.mean([u.G for u in population])
        mean_l = np.mean([u.Lambda for u in population])
        max_fit = np.max(scores)
        
        avg_g_history.append(mean_g)
        avg_l_history.append(mean_l)
        best_fitness_history.append(max_fit)
        
        if gen % 10 == 0:
            print(f"Gen {gen}: Mean G={mean_g:.2f}, Mean Lambda={mean_l:.2f}, Max BHs={max_fit:.0f}")
        
        # B. Selection (Tournament / Rank)
        # Sort by fitness (descending)
        sorted_indices = np.argsort(scores)[::-1]
        survivors = [population[i] for i in sorted_indices[:int(pop_size*0.2)]] # Top 20%
        
        # C. Reproduction (Crossover + Mutation)
        new_population = []
        
        # Elitism (Keep top 5 unchanged)
        new_population.extend(survivors[:5])
        
        while len(new_population) < pop_size:
            # Pick parents
            parent_a = np.random.choice(survivors)
            parent_b = np.random.choice(survivors)
            
            # Crossover
            child_g = (parent_a.G + parent_b.G) / 2.0
            child_l = (parent_a.Lambda + parent_b.Lambda) / 2.0
            child_s = (parent_a.Sigma + parent_b.Sigma) / 2.0
            
            # Mutation (Small drift)
            if np.random.random() < 0.3:
                child_g += np.random.normal(0, 0.1)
            if np.random.random() < 0.3:
                child_l += np.random.normal(0, 0.1)
                
            # Clamp limits
            child_g = max(0.1, child_g)
            child_l = max(0.0, child_l)
            
            new_population.append(Universe(child_g, child_l, child_s))
            
        population = new_population

    # 3. Final Analysis
    winner = population[0] # Best of final gen (since we sorted/elitism)
    
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print(f"Winner Parameters:")
    print(f"  G      = {winner.G:.2f} (Ours: {OUR_G})")
    print(f"  Lambda = {winner.Lambda:.2f} (Ours: {OUR_LAMBDA})")
    print(f"  Sigma  = {winner.Sigma:.2f} (Ours: {OUR_SIGMA})")
    print(f"  Fitness: {winner.fitness:.0f} Black Holes")
    print("-" * 30)
    
    # Check for "Fine Tuning"
    # Does the winner match "Us"?
    
    g_diff = abs(winner.G - OUR_G)
    l_diff = abs(winner.Lambda - OUR_LAMBDA)
    
    match = False
    if g_diff < 0.3 and l_diff < 0.3:
        match = True
        
    print("VERDICT: COSMOLOGICAL SELECTION")
    if match:
        print("RESULT: POSITIVE (+)")
        print("The Simulation CONVERGED on our universe's parameters!")
        print("Our physics are the result of evolutionary optimization.")
    else:
        print("RESULT: MIXED / DIVERGENT")
        print("The simulation found a different optimization peak.")
        print(f"It prefers G={winner.G:.2f}, L={winner.Lambda:.2f}.")
    print("="*60)
    
    # 4. Plot Evolution
    plt.figure(figsize=(12, 5), facecolor='black')
    
    ax1 = plt.subplot(1, 2, 1, facecolor='#1a1a2e')
    ax1.plot(avg_g_history, 'c-', label='Gravity (G)')
    ax1.plot(avg_l_history, 'm-', label='Dark Energy (Lambda)')
    ax1.axhline(OUR_G, color='c', linestyle='--', alpha=0.5, label='Our G')
    ax1.axhline(OUR_LAMBDA, color='m', linestyle='--', alpha=0.5, label='Our Lambda')
    ax1.set_xlabel('Generation', color='white')
    ax1.set_ylabel('Parameter Value', color='white')
    ax1.set_title('Evolution of Cosmic Constants', color='white')
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax1.tick_params(colors='white')
    
    ax2 = plt.subplot(1, 2, 2, facecolor='#1a1a2e')
    ax2.plot(best_fitness_history, 'y-')
    ax2.set_xlabel('Generation', color='white')
    ax2.set_ylabel('Max Black Holes (Fitness)', color='white')
    ax2.set_title('Fitness Landscape', color='white')
    ax2.grid(alpha=0.2)
    ax2.tick_params(colors='white')
    
    plt.savefig('evolution_results.png', dpi=100, facecolor='black')
    print("   Saved plot to evolution_results.png")

if __name__ == "__main__":
    run_tournament()
