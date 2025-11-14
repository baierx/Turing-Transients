"""
Configuration file for PSO parameter estimation
Adjust these parameters to control the optimization process
"""

# PSO Parameters
PSO_CONFIG = {
    'n_particles': 30,        # Number of particles in swarm (20-40 recommended)
    'max_iterations': 50,     # Maximum iterations (50-100 recommended)
    'w': 0.7,                 # Inertia weight (0.4-0.9, controls momentum)
    'c1': 1.5,                # Cognitive parameter (1.5-2.0, personal best influence)
    'c2': 1.5,                # Social parameter (1.5-2.0, global best influence)
}

# Parameter bounds to search
PARAMETER_BOUNDS = {
    'frac_EE': (0.001, 0.5),      # Excitatory coupling fraction
    'pulse_fre': (0.001, 1.0),    # Pulse frequency
    'pulse_amp': (0.001, 5.0),    # Pulse amplitude
}

# Simulation parameters
SIMULATION_CONFIG = {
    'L': 3,                   # Number of hexagonal layers
    'time_stop': 30,          # Simulation duration (seconds)
    'sr': 1000,               # Sampling rate (Hz)
}

# Threshold for identifying successful wave propagation
THRESHOLD = 0.0               # Activity must exceed this value in outer layer

# Output settings
OUTPUT_CONFIG = {
    'save_plots': True,
    'save_results': True,
    'plot_convergence': True,
    'plot_parameter_space': True,
}

# Quick test mode (for debugging)
QUICK_TEST = False
if QUICK_TEST:
    PSO_CONFIG['n_particles'] = 10
    PSO_CONFIG['max_iterations'] = 5
    SIMULATION_CONFIG['time_stop'] = 10
