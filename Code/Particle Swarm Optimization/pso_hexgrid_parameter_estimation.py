"""
Particle Swarm Optimization for Parameter Estimation of Wave Propagation on Hexagonal Grid
Finds parameter combinations (frac_EE, pulse_fre, pulse_amp) where activity in outer layer exceeds threshold
"""

from scipy.integrate import odeint
from numpy import zeros, ones, tanh, mod, gradient, linspace, sign, log, meshgrid, sort, pi, exp, array, around, arange
from numpy import sqrt, fill_diagonal, ndarray, amax, amin, where, c_, histogram, complex128, flip, ravel_multi_index, arctan2
from numpy.random import normal, seed, rand, uniform
import sk_dsp_comm.sigsys as ss
import numpy as np
from time import time as timer
import matplotlib.pyplot as plt

def sigmoid(u):
    return tanh(u)

def N_oscillators(y, t, N, h_ex_rand, h_in_rand,
                  coupling_matrix_EE, coupling_matrix_EI, coupling_strength_EE, coupling_strength_EI, pars, sr, time_stop, pert, pert_osc):
    tau_ex, tau_in, c2, c4 = pars
    
    time_index = int(t*sr)
    
    if time_index >= time_stop*sr:
        dydt = zeros(2*N)
        return dydt
    
    # Separate Variables
    y_ex = y[:-1:2]
    y_in = y[1::2]
    
    dy_ex, dy_in = zeros(N), zeros(N)
    dydt = zeros(2*N)
    
    for osc in arange(N):
        if osc == pert_osc:
            coup_EE = sum(coupling_matrix_EE[:, osc] * y_ex)
            coup_EI = sum(coupling_matrix_EI[:, osc] * y_ex)
                
            dy_ex[osc] = (pert[time_index] - y_ex[osc] - c2*sigmoid(y_in[osc]) + 
                          coupling_strength_EE*sigmoid(coup_EE))*tau_ex 
            dy_in[osc] = (h_in_rand[osc]   - y_in[osc] - c4*sigmoid(y_in[osc]) + 
                          coupling_strength_EI*sigmoid(coup_EI))*tau_in
        else:
            coup_EE = sum(coupling_matrix_EE[:, osc] * y_ex)
            coup_EI = sum(coupling_matrix_EI[:, osc] * y_ex)
                
            dy_ex[osc] = (h_ex_rand[osc] - y_ex[osc] - c2*sigmoid(y_in[osc]) + 
                          coupling_strength_EE*sigmoid(coup_EE))*tau_ex 
            dy_in[osc] = (h_in_rand[osc] - y_in[osc] - c4*sigmoid(y_in[osc]) + 
                          coupling_strength_EI*sigmoid(coup_EI))*tau_in
    
    # Combine Variables
    dydt[:-1:2] = dy_ex
    dydt[1: :2] = dy_in
    
    return dydt

def get_hexagon_centers_sorted(layers):
    """
    Generate hexagon centers sorted by layer (distance from origin)
    Returns list of (x, y, q, r, s, distance) tuples
    """
    center_spacing = 1.0
    centers = []
    for q in range(-layers+1, layers):
        for r in range(-layers+1, layers):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= layers-1:
                x = center_spacing * (3/2 * q)
                y = center_spacing * (sqrt(3)/2 * q + sqrt(3) * r)
                distance = max(abs(q), abs(r), abs(s))
                centers.append((x, y, q, r, s, distance))
    
    # Sort by distance (layer), then by angle
    centers_sorted = sorted(centers, key=lambda c: (c[5], arctan2(c[1], c[0])))
    return centers_sorted

def create_connectivity_matrix(layers):
    """
    Create connectivity/adjacency matrix for hexagonal lattice
    Returns NxN matrix where entry (i,j) = 1 if hexagons i and j are neighbors
    """
    centers = get_hexagon_centers_sorted(layers)
    N = len(centers)
    # Initialize connectivity matrix
    connectivity = zeros((N, N), dtype=int)
    
    # Create a mapping from (q, r, s) coordinates to index
    coord_to_index = {}
    for i, (x, y, q, r, s, dist) in enumerate(centers):
        coord_to_index[(q, r, s)] = i
    
    neighbor_offsets = [
        (1, -1, 0),   # East
        (1, 0, -1),   # Northeast
        (0, 1, -1),   # Northwest
        (-1, 1, 0),   # West
        (-1, 0, 1),   # Southwest
        (0, -1, 1)    # Southeast
    ]
    
    # For each hexagon, check its 6 potential neighbors
    for i, (x, y, q, r, s, dist) in enumerate(centers):
        for dq, dr, ds in neighbor_offsets:
            neighbor_coords = (q + dq, r + dr, s + ds)
            
            # Check if neighbor exists in our lattice
            if neighbor_coords in coord_to_index:
                j = coord_to_index[neighbor_coords]
                connectivity[i, j] = 1
    
    return connectivity, centers


def simulate_hexgrid(frac_EE, pulse_fre, pulse_amp, L=3, time_stop=30, sr=1000, verbose=False):
    """
    Run simulation with given parameters
    Returns maximum value in outer layer (last 12 oscillators) during last 10000 time points
    """
    # Number of oscillators
    N = 3*L*(L-1)+1
    
    # Excitatory input parameter
    h_ex_0 = -6.4
    h_in_0 = -4.0
    eps = 0.01
    RANDOM_STATE = 11111
    seed(RANDOM_STATE)
    random_vals = eps*normal(0, 1, size=N)
    random_vals_sorted = sort(random_vals)
    h_ex_rand = h_ex_0 - random_vals_sorted
    h_in_rand = h_in_0 - eps*normal(0, 1, size=N)
    
    # Parameters
    pars = (1, 1.5, 10, 0)
    coupling_strength_EE, coupling_strength_EI = 5., 10.
    frac_EI = 0.0
    
    # Coupling
    coupling_matrix_EE_ini, _ = create_connectivity_matrix(L)
    coupling_matrix_EI_ini = coupling_matrix_EE_ini.copy()
    coupling_matrix_EE = coupling_matrix_EE_ini*frac_EE
    coupling_matrix_EI = coupling_matrix_EI_ini*frac_EI
    fill_diagonal(coupling_matrix_EE, 1)
    fill_diagonal(coupling_matrix_EI, 1)
    
    # Time array
    samples = time_stop*sr
    time = linspace(start=0, stop=time_stop, num=samples)
    pulse_wid = 0.2
    
    # Initial conditions
    y_ini = normal(size=2*N)
    pert = h_ex_0 + pulse_amp*ss.rect(mod(time, 1/pulse_fre)-(1/pulse_fre)/2-pulse_wid/2, pulse_wid)
    pert_osc = 0
    
    # Simulation
    try:
        y_pert = odeint(func=N_oscillators, y0=y_ini, t=time,
                       args=(N, h_ex_rand, h_in_rand,
                             coupling_matrix_EE,
                             coupling_matrix_EI,
                             coupling_strength_EE,
                             coupling_strength_EI,
                             pars, sr, time_stop, pert, pert_osc),
                       hmax=0.1)
        y_ex_only = y_pert[:, ::2]
        
        # Check outer layer (last 12 oscillators) in last 10000 time points
        outer_layer_activity = y_ex_only[20000:, -12:]
        max_activity = amax(outer_layer_activity)
        
        if verbose:
            print(f"  frac_EE={frac_EE:.3f}, pulse_fre={pulse_fre:.3f}, pulse_amp={pulse_amp:.3f} -> max={max_activity:.3f}")
        
        return max_activity
    
    except Exception as e:
        if verbose:
            print(f"  Simulation failed: {e}")
        return -1e10  # Return large negative value for failed simulations


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for finding parameter combinations
    """
    def __init__(self, bounds, n_particles=30, max_iterations=50, w=0.7, c1=1.5, c2=1.5):
        """
        bounds: list of (min, max) tuples for each parameter
        n_particles: number of particles in swarm
        max_iterations: maximum number of iterations
        w: inertia weight
        c1: cognitive parameter (personal best)
        c2: social parameter (global best)
        """
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_dims = len(bounds)
        
        # Initialize particles
        self.positions = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            (n_particles, self.n_dims)
        )
        
        # Initialize velocities
        velocity_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range,
            velocity_range,
            (n_particles, self.n_dims)
        )
        
        # Best positions
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(n_particles, -np.inf)
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        # History
        self.history = {
            'global_best_scores': [],
            'mean_scores': [],
            'all_positions': [],
            'all_scores': []
        }
    
    def optimize(self, objective_function, verbose=True):
        """
        Run PSO optimization
        objective_function: function that takes parameter array and returns fitness score
        """
        start_time = timer()
        
        for iteration in range(self.max_iterations):
            iter_start = timer()
            scores = np.zeros(self.n_particles)
            
            # Evaluate all particles
            for i in range(self.n_particles):
                scores[i] = objective_function(self.positions[i])
                
                # Update personal best
                if scores[i] > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                # Update global best
                if scores[i] > self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()
            
            # Store history
            self.history['global_best_scores'].append(self.global_best_score)
            self.history['mean_scores'].append(np.mean(scores))
            self.history['all_positions'].append(self.positions.copy())
            self.history['all_scores'].append(scores.copy())
            
            iter_time = timer() - iter_start
            
            if verbose:
                print(f"Iteration {iteration+1}/{self.max_iterations}: "
                      f"Best={self.global_best_score:.4f}, "
                      f"Mean={np.mean(scores):.4f}, "
                      f"Time={iter_time:.2f}s")
            
            # Update velocities and positions
            r1 = np.random.random((self.n_particles, self.n_dims))
            r2 = np.random.random((self.n_particles, self.n_dims))
            
            cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social = self.c2 * r2 * (self.global_best_position - self.positions)
            
            self.velocities = self.w * self.velocities + cognitive + social
            
            # Update positions
            self.positions = self.positions + self.velocities
            
            # Apply bounds
            self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])
        
        total_time = timer() - start_time
        
        if verbose:
            print(f"\nOptimization complete in {total_time:.2f}s")
            print(f"Best parameters found: frac_EE={self.global_best_position[0]:.4f}, "
                  f"pulse_fre={self.global_best_position[1]:.4f}, "
                  f"pulse_amp={self.global_best_position[2]:.4f}")
            print(f"Best fitness: {self.global_best_score:.4f}")
        
        return self.global_best_position, self.global_best_score
    
    def get_all_threshold_exceeding_params(self, threshold=0.0):
        """
        Get all parameter combinations that exceeded the threshold
        Returns array of (frac_EE, pulse_fre, pulse_amp, fitness) for all valid combinations
        """
        valid_params = []
        
        for positions, scores in zip(self.history['all_positions'], self.history['all_scores']):
            for pos, score in zip(positions, scores):
                if score > threshold:
                    valid_params.append((*pos, score))
        
        return np.array(valid_params) if valid_params else np.array([])
    
    def plot_convergence(self, save_path=None):
        """Plot PSO convergence"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = range(1, len(self.history['global_best_scores']) + 1)
        
        # Plot best and mean fitness
        ax1.plot(iterations, self.history['global_best_scores'], 'b-', linewidth=2, label='Global Best')
        ax1.plot(iterations, self.history['mean_scores'], 'r--', linewidth=1.5, label='Mean')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Fitness (Max Activity)', fontsize=12)
        ax1.set_title('PSO Convergence', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot parameter evolution
        all_positions = np.array(self.history['all_positions'])
        all_scores = np.array(self.history['all_scores'])
        
        # Scatter plot of last iteration colored by fitness
        last_positions = all_positions[-1]
        last_scores = all_scores[-1]
        
        scatter = ax2.scatter(last_positions[:, 0], last_positions[:, 1], 
                             c=last_scores, s=100, alpha=0.6, cmap='viridis')
        ax2.scatter(self.global_best_position[0], self.global_best_position[1],
                   c='red', s=200, marker='*', edgecolors='black', linewidths=2,
                   label='Global Best', zorder=5)
        ax2.set_xlabel('frac_EE', fontsize=12)
        ax2.set_ylabel('pulse_fre', fontsize=12)
        ax2.set_title('Final Particle Distribution (colored by fitness)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Fitness', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_space(self, threshold=0.0, save_path=None):
        """
        Plot 3D parameter space showing all explored points
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Collect all explored parameters
        all_params = []
        for positions, scores in zip(self.history['all_positions'], self.history['all_scores']):
            for pos, score in zip(positions, scores):
                all_params.append((*pos, score))
        
        all_params = np.array(all_params)
        
        # Separate by threshold
        valid_mask = all_params[:, 3] > threshold
        valid_params = all_params[valid_mask]
        invalid_params = all_params[~valid_mask]
        
        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        if len(invalid_params) > 0:
            ax1.scatter(invalid_params[:, 0], invalid_params[:, 1], invalid_params[:, 2],
                       c='lightgray', alpha=0.3, s=20, label='Below threshold')
        if len(valid_params) > 0:
            scatter = ax1.scatter(valid_params[:, 0], valid_params[:, 1], valid_params[:, 2],
                                 c=valid_params[:, 3], cmap='hot', alpha=0.6, s=50, 
                                 label='Above threshold')
            plt.colorbar(scatter, ax=ax1, label='Fitness')
        
        ax1.scatter(self.global_best_position[0], self.global_best_position[1], 
                   self.global_best_position[2], c='blue', s=200, marker='*',
                   edgecolors='black', linewidths=2, label='Global Best')
        
        ax1.set_xlabel('frac_EE', fontsize=10)
        ax1.set_ylabel('pulse_fre', fontsize=10)
        ax1.set_zlabel('pulse_amp', fontsize=10)
        ax1.set_title('3D Parameter Space', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        
        # 2D projections
        ax2 = fig.add_subplot(132)
        if len(invalid_params) > 0:
            ax2.scatter(invalid_params[:, 0], invalid_params[:, 1], c='lightgray', alpha=0.3, s=20)
        if len(valid_params) > 0:
            ax2.scatter(valid_params[:, 0], valid_params[:, 1], 
                       c=valid_params[:, 3], cmap='hot', alpha=0.6, s=50)
        ax2.scatter(self.global_best_position[0], self.global_best_position[1],
                   c='blue', s=200, marker='*', edgecolors='black', linewidths=2)
        ax2.set_xlabel('frac_EE', fontsize=10)
        ax2.set_ylabel('pulse_fre', fontsize=10)
        ax2.set_title('frac_EE vs pulse_fre', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(133)
        if len(invalid_params) > 0:
            ax3.scatter(invalid_params[:, 0], invalid_params[:, 2], c='lightgray', alpha=0.3, s=20)
        if len(valid_params) > 0:
            ax3.scatter(valid_params[:, 0], valid_params[:, 2],
                       c=valid_params[:, 3], cmap='hot', alpha=0.6, s=50)
        ax3.scatter(self.global_best_position[0], self.global_best_position[2],
                   c='blue', s=200, marker='*', edgecolors='black', linewidths=2)
        ax3.set_xlabel('frac_EE', fontsize=10)
        ax3.set_ylabel('pulse_amp', fontsize=10)
        ax3.set_title('frac_EE vs pulse_amp', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter space plot saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Parameter Space Exploration Results")
        print(f"{'='*60}")
        print(f"Total evaluations: {len(all_params)}")
        print(f"Above threshold (>{threshold}): {len(valid_params)}")
        print(f"Below threshold: {len(invalid_params)}")
        if len(valid_params) > 0:
            print(f"\nValid parameter ranges:")
            print(f"  frac_EE:   [{valid_params[:, 0].min():.4f}, {valid_params[:, 0].max():.4f}]")
            print(f"  pulse_fre: [{valid_params[:, 1].min():.4f}, {valid_params[:, 1].max():.4f}]")
            print(f"  pulse_amp: [{valid_params[:, 2].min():.4f}, {valid_params[:, 2].max():.4f}]")
            print(f"  fitness:   [{valid_params[:, 3].min():.4f}, {valid_params[:, 3].max():.4f}]")


def main():
    """
    Main function to run PSO for hexagonal grid parameter estimation
    """
    print("="*60)
    print("Particle Swarm Optimization for Hexagonal Grid Wave Propagation")
    print("="*60)
    print("\nParameter ranges:")
    print("  frac_EE:   (0.0, 0.5)")
    print("  pulse_fre: (0.0, 1.0)")
    print("  pulse_amp: (0.0, 5.0)")
    print("\nObjective: Find parameters where outer layer activity exceeds 0")
    print("="*60)
    
    # Define parameter bounds
    bounds = [
        (0.001, 0.5),    # frac_EE (avoid exactly 0)
        (0.001, 1.0),    # pulse_fre (avoid exactly 0)
        (0.001, 5.0)     # pulse_amp (avoid exactly 0)
    ]
    
    # Define objective function
    def objective(params):
        frac_EE, pulse_fre, pulse_amp = params
        return simulate_hexgrid(frac_EE, pulse_fre, pulse_amp)
    
    # Initialize PSO
    pso = ParticleSwarmOptimizer(
        bounds=bounds,
        n_particles=30,      # Standard: 20-40 particles
        max_iterations=50,   # Standard: 50-100 iterations
        w=0.7,              # Inertia weight (0.4-0.9)
        c1=1.5,             # Cognitive parameter (1.5-2.0)
        c2=1.5              # Social parameter (1.5-2.0)
    )
    
    # Run optimization
    print("\nStarting PSO optimization...\n")
    best_params, best_fitness = pso.optimize(objective, verbose=True)
    
    # Get all parameter combinations that exceeded threshold
    threshold_params = pso.get_all_threshold_exceeding_params(threshold=0.0)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Number of parameter combinations exceeding threshold: {len(threshold_params)}")
    
    if len(threshold_params) > 0:
        print("\nTop 10 parameter combinations:")
        print(f"{'frac_EE':<12} {'pulse_fre':<12} {'pulse_amp':<12} {'fitness':<12}")
        print("-" * 50)
        
        # Sort by fitness
        sorted_params = threshold_params[threshold_params[:, 3].argsort()[::-1]]
        for i, params in enumerate(sorted_params[:10]):
            print(f"{params[0]:<12.4f} {params[1]:<12.4f} {params[2]:<12.4f} {params[3]:<12.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    pso.plot_convergence(save_path='/mnt/user-data/outputs/pso_convergence.png')
    pso.plot_parameter_space(threshold=0.0, save_path='/mnt/user-data/outputs/pso_parameter_space.png')
    
    # Save results to file
    results_file = '/mnt/user-data/outputs/pso_results.txt'
    with open(results_file, 'w') as f:
        f.write("PSO Parameter Estimation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best parameters found:\n")
        f.write(f"  frac_EE:   {best_params[0]:.6f}\n")
        f.write(f"  pulse_fre: {best_params[1]:.6f}\n")
        f.write(f"  pulse_amp: {best_params[2]:.6f}\n")
        f.write(f"  fitness:   {best_fitness:.6f}\n\n")
        
        f.write(f"Parameter combinations exceeding threshold (>0):\n")
        f.write(f"Total: {len(threshold_params)}\n\n")
        
        if len(threshold_params) > 0:
            f.write(f"{'frac_EE':<12} {'pulse_fre':<12} {'pulse_amp':<12} {'fitness':<12}\n")
            f.write("-" * 50 + "\n")
            for params in sorted_params:
                f.write(f"{params[0]:<12.6f} {params[1]:<12.6f} {params[2]:<12.6f} {params[3]:<12.6f}\n")
    
    print(f"\nResults saved to {results_file}")
    print("\nOptimization complete!")
    
    return pso, best_params, best_fitness, threshold_params


if __name__ == "__main__":
    pso, best_params, best_fitness, threshold_params = main()
