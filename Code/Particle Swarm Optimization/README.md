# Particle Swarm Optimization for Hexagonal Grid Parameter Estimation

## Overview

This PSO implementation finds parameter combinations (frac_EE, pulse_fre, pulse_amp) that produce wave propagation exceeding a threshold in the outer layer of a hexagonal lattice network.

## Files

1. **run_pso_simple.py** - Main script (recommended for most users)
   - Complete, standalone implementation
   - Easy to modify parameters directly in the code
   - Generates plots and saves results

2. **pso_hexgrid_parameter_estimation.py** - Full-featured version
   - More detailed class structure
   - Additional plotting and analysis features
   - Modular design for advanced users

3. **pso_config.py** - Configuration file
   - Adjust PSO parameters without modifying main code
   - Quick test mode for debugging

## Quick Start

### Basic Usage

```bash
python run_pso_simple.py
```

This will:
- Run PSO with 30 particles for 50 iterations
- Search the parameter space:
  - frac_EE: (0.001, 0.5)
  - pulse_fre: (0.001, 1.0)
  - pulse_amp: (0.001, 5.0)
- Find parameters where outer layer activity exceeds 0
- Generate visualization plots
- Save results to text file

### Expected Runtime

- ~2-5 minutes per iteration (30 simulations)
- Total: ~2-4 hours for 50 iterations
- Quick test mode: ~5-10 minutes

## Understanding the Output

### Fitness Function

The fitness is the **maximum activity** in the outer layer (last 12 oscillators) during the last 10,000 time points (20,000:30,000).

- **Fitness > 0**: Wave successfully propagated to outer layer
- **Fitness ≤ 0**: Wave did not reach outer layer
- **Higher fitness**: Stronger propagation

### Console Output

```
Iter 1/50: Best=2.3456, Mean=0.1234, Time=3.2s/iter
```

- **Best**: Best fitness found so far across all particles
- **Mean**: Average fitness of current generation
- **Time**: Average time per iteration

### Results Files

1. **pso_results.png** - Visualization with three panels:
   - Convergence plot (best and mean fitness over iterations)
   - 3D parameter space (all explored points)
   - 2D projection (frac_EE vs pulse_fre)

2. **pso_all_results.txt** - All parameter combinations exceeding threshold
   - Sorted by fitness (highest first)
   - Full precision parameter values

## Customizing the Search

### Adjusting PSO Parameters

Edit in `run_pso_simple.py` (line ~315):

```python
pso = PSO(bounds=bounds, 
          n_particles=30,    # More particles = better coverage, slower
          max_iter=50,       # More iterations = better convergence
          w=0.7,            # Inertia (0.4-0.9): higher = more exploration
          c1=1.5,           # Personal best weight (1.5-2.0)
          c2=1.5)           # Global best weight (1.5-2.0)
```

**Guidelines:**
- **n_particles**: 20-40 is standard (30 is good default)
- **max_iter**: 50-100 is standard (more for complex landscapes)
- **w**: 0.7-0.9 for exploration, 0.4-0.6 for exploitation
- **c1, c2**: Usually equal, around 1.5-2.0

### Adjusting Parameter Bounds

Edit in `run_pso_simple.py` (line ~310):

```python
bounds = [
    (0.001, 0.5),    # frac_EE
    (0.001, 1.0),    # pulse_fre
    (0.001, 5.0)     # pulse_amp
]
```

### Changing Simulation Parameters

Edit in `simulate_hexgrid()` function:

```python
def simulate_hexgrid(frac_EE, pulse_fre, pulse_amp, 
                     L=3,           # Hexagonal layers
                     time_stop=30,  # Simulation duration (s)
                     sr=1000):      # Sampling rate (Hz)
```

### Changing Threshold

Edit in `main()` function (line ~345):

```python
threshold_params = pso.get_threshold_exceeding(threshold=0.0)
```

Change `threshold=0.0` to any value you want.

## Quick Test Mode

For fast testing (10 particles, 5 iterations, 10s simulation):

Edit at the top of `run_pso_simple.py`:

```python
# Quick test for debugging
QUICK_TEST = True

if QUICK_TEST:
    # ... (modify parameters in main())
    pso = PSO(bounds=bounds, n_particles=10, max_iter=5)
    # ... and in simulate_hexgrid:
    return simulate_hexgrid(params[0], params[1], params[2], time_stop=10)
```

## Interpreting Results

### Convergence Behavior

**Good convergence:**
- Best fitness improves steadily
- Mean fitness increases over time
- Particles cluster around good regions

**Poor convergence:**
- Best fitness plateaus early
- Large gap between best and mean
- May need more iterations or different w/c1/c2

### Parameter Space

**Well-explored space:**
- Points distributed across entire range
- Clear clusters of high-fitness regions
- Smooth gradients visible

**Under-explored space:**
- Sparse point distribution
- Large gaps in coverage
- May need more particles or iterations

## Advanced Features

### Accessing PSO Object

After running, you can access the PSO object for further analysis:

```python
pso, best_params, best_fitness, threshold_params = main()

# Get all explored positions
all_positions = pso.history['positions']
all_scores = pso.history['scores']

# Get convergence history
convergence = pso.history['g_best']

# Get personal bests of all particles
personal_bests = pso.p_best_pos
personal_best_scores = pso.p_best_scores
```

### Running Multiple Optimizations

For better coverage, run multiple independent PSO runs:

```python
results = []
for run in range(5):
    pso = PSO(bounds=bounds, n_particles=30, max_iter=50)
    best_params, best_fitness = pso.optimize(objective)
    results.append((best_params, best_fitness))
    print(f"Run {run+1}: Best fitness = {best_fitness:.4f}")

# Find overall best
best_run = max(results, key=lambda x: x[1])
print(f"Overall best: {best_run}")
```

## Troubleshooting

### Simulations failing (fitness = -1e10)

**Cause**: ODE solver encountering numerical issues

**Solutions:**
- Reduce `hmax` in `odeint` call (currently 0.1)
- Reduce parameter ranges to avoid extreme values
- Check initial conditions

### Very slow execution

**Causes:**
- Long simulation time (30s with sr=1000 = 30,000 points)
- Complex ODE system (19 oscillators × 2 variables = 38 ODEs)

**Solutions:**
- Reduce `time_stop` (e.g., 20s instead of 30s)
- Reduce `sr` (e.g., 500 instead of 1000) - but check if results still meaningful
- Use quick test mode for initial exploration
- Run on faster hardware or reduce PSO parameters

### PSO not converging

**Solutions:**
- Increase `max_iter` (try 100)
- Adjust inertia weight `w` (try 0.5-0.9)
- Increase `n_particles` (try 40-50)
- Check if fitness landscape is too flat (most values near zero)

### No parameters exceed threshold

**Solutions:**
- Lower threshold (try -0.5 or -1.0)
- Expand parameter ranges
- Check if simulation is working correctly (test with known good parameters)
- Increase PSO exploration (higher `w`, more particles)

## Performance Tips

1. **Start with quick test**: Use 10 particles, 5 iterations to verify setup
2. **Profile parameter space**: Run grid search on coarse grid first to identify promising regions
3. **Warm start**: Initialize particles near known good parameters instead of random
4. **Parallel evaluation**: Modify code to evaluate particles in parallel (requires multiprocessing)
5. **Adaptive parameters**: Decrease `w` over iterations for exploration→exploitation transition

## Citation

If you use this code for your research, please cite:

```
Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. 
Proceedings of ICNN'95-International Conference on Neural Networks.
```

## Contact

For questions about the hexagonal grid model or this implementation, please refer to your original simulation code documentation.
