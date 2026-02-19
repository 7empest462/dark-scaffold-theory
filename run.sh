#!/bin/bash
# Runner script for Dark Scaffold cosmology simulations
# Ensures the correct Python environment is used

PYTHON="/Users/7empest_mac/miniforge3/envs/cosmology_new/bin/python"
PROJECT_DIR="/Volumes/Corsair_Lab/Home/Documents/Cosmology/dark-scaffold-theory"

cd "$PROJECT_DIR"

case "$1" in
    scaffold)
        echo "Generating dark matter scaffold..."
        $PYTHON scaffold_generator.py
        ;;
    seeping)
        echo "Running seeping simulation..."
        $PYTHON seeping_simulation.py
        ;;
    nbody)
        echo "Running N-body simulation..."
        $PYTHON nbody_simulation.py
        ;;
    infall)
        echo "Running accelerated infall simulation..."
        $PYTHON infall_simulation.py
        ;;
    energy)
        echo "Running energy budget analysis..."
        $PYTHON energy_budget.py
        ;;
    likelihood)
        echo "Running likelihood assessment..."
        $PYTHON theory_likelihood.py
        ;;
    optimize)
        echo "Running parameter optimization..."
        $PYTHON theory_optimizer.py
        ;;
    all)
        echo "Running full analysis pipeline..."
        $PYTHON scaffold_generator.py
        $PYTHON energy_budget.py
        $PYTHON theory_likelihood.py
        ;;
    *)
        echo "Dark Scaffold Cosmology Runner"
        echo ""
        echo "Usage: $0 {scaffold|seeping|energy|likelihood|optimize|all}"
        echo ""
        echo "Commands:"
        echo "  scaffold   - Generate dark matter scaffold"
        echo "  seeping    - Run seeping simulation"
        echo "  energy     - Run energy budget analysis"
        echo "  likelihood - Run likelihood assessment"
        echo "  optimize   - Run parameter optimization"
        echo "  all        - Run full analysis pipeline"
        ;;
esac
