"""
Simple PSO Runner for Hexagonal Grid Parameter Estimation
Usage: python run_pso_simple.py
"""

from scipy.integrate import odeint
from numpy import zeros, ones, tanh, mod, linspace, sort, pi, exp, array, arange
from numpy import sqrt, fill_diagonal, amax, arctan2
from numpy.random import normal, seed
import sk_dsp_comm.sigsys as ss
import numpy as np
from time import time as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# CORE SIMULATION FUNCTIONS
# ============================================================================

def sigmoid(u):
    return tanh(u)

def N_oscillators(y, t, N, h_ex_rand, h_in_rand,
                  coupling_matrix_EE, coupling_matrix_EI, 
                  coupling_strength_EE, coupling_strength_EI, 
                  pars, sr, time_stop, pert, pert_osc):
    tau_ex, tau_in, c2, c4 = pars
    time_index = int(t*sr)
    
    if time_index >= time_stop*sr:
        return zeros(2*N)
    
    y_ex = y[:-1:2]
    y_in = y[1::2]
    dy_ex, dy_in = zeros(N), zeros(N)
    dydt = zeros(2*N)
    
    for osc in arange(N):
        coup_EE = sum(coupling_matrix_EE[:, osc] * y_ex)
        coup_EI = sum(coupling_matrix_EI[:, osc] * y_ex)
        
        if osc == pert_osc:
            dy_ex[osc] = (pert[time_index] - y_ex[osc] - c2*sigmoid(y_in[osc]) + 
                          coupling_strength_EE*sigmoid(coup_EE))*tau_ex 
        else:
            dy_ex[osc] = (h_ex_rand[osc] - y_ex[osc] - c2*sigmoid(y_in[osc]) + 
                          coupling_strength_EE*sigmoid(coup_EE))*tau_ex 
        
        dy_in[osc] = (h_in_rand[osc] - y_in[osc] - c4*sigmoid(y_in[osc]) + 
                      coupling_strength_EI*sigmoid(coup_EI))*tau_in
    
    dydt[:-1:2] = dy_ex
    dydt[1::2] = dy_in
    return dydt

def get_hexagon_centers_sorted(layers):
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
    
    centers_sorted = sorted(centers, key=lambda c: (c[5], arctan2(c[1], c[0])))
    return centers_sorted

def create_connectivity_matrix(layers):
    centers = get_hexagon_centers_sorted(layers)
    N = len(centers)
    connectivity = zeros((N, N), dtype=int)
    
    coord_to_index = {}
    for i, (x, y, q, r, s, dist) in enumerate(centers):
        coord_to_index[(q, r, s)] = i
    
    neighbor_offsets = [
        (1, -1, 0), (1, 0, -1), (0, 1, -1),
        (-1, 1, 0), (-1, 0, 1), (0, -1, 1)
    ]
    
    for i, (x, y, q, r, s, dist) in enumerate(centers):
        for dq, dr, ds in neighbor_offsets:
            neighbor_coords = (q + dq, r + dr, s + ds)
            if neighbor_coords in coord_to_index:
                j = coord_to_index[neighbor_coords]
                connectivity[i, j] = 1
    
    return connectivity, centers

def simulate_hexgrid(frac_EE, pulse_fre, pulse_amp, L=3, time_stop=30, sr=1000):
    """Run simulation and return max activity in outer layer"""
    N = 3*L*(L-1)+1
    
    # Fixed parameters
    h_ex_0, h_in_0, eps = -6.4, -4.0, 0.01
    RANDOM_STATE = 11111
    seed(RANDOM_STATE)
    
    random_vals = eps*normal(0, 1, size=N)
    random_vals_sorted = sort(random_vals)
    h_ex_rand = h_ex_0 - random_vals_sorted
    h_in_rand = h_in_0 - eps*normal(0, 1, size=N)
    
    pars = (1, 1.5, 10, 0)
    coupling_strength_EE, coupling_strength_EI = 5., 10.
    frac_EI = 0.0
    
    coupling_matrix_EE_ini, _ = create_connectivity_matrix(L)
    coupling_matrix_EI_ini = coupling_matrix_EE_ini.copy()
    coupling_matrix_EE = coupling_matrix_EE_ini*frac_EE
    coupling_matrix_EI = coupling_matrix_EI_ini*frac_EI
    fill_diagonal(coupling_matrix_EE, 1)
    fill_diagonal(coupling_matrix_EI, 1)
    
    samples = time_stop*sr
    time = linspace(start=0, stop=time_stop, num=samples)
    pulse_wid = 0.2
    
    y_ini = normal(size=2*N)
    pert = h_ex_0 + pulse_amp*ss.rect(mod(time, 1/pulse_fre)-(1/pulse_fre)/2-pulse_wid/2, pulse_wid)
    pert_osc = 0
    
    try:
        y_pert = odeint(func=N_oscillators, y0=y_ini, t=time,
                       args=(N, h_ex_rand, h_in_rand,
                             coupling_matrix_EE, coupling_matrix_EI,
                             coupling_strength_EE, coupling_strength_EI,
                             pars, sr, time_stop, pert, pert_osc),
                       hmax=0.1)
        y_ex_only = y_pert[:, ::2]
        
        # Check outer layer (last 12 oscillators) in last 10000 time points
        outer_layer_activity = y_ex_only[20000:, -12:]
        max_activity = amax(outer_layer_activity)
        
        return max_activity
    except:
        return -1e10  # Return large negative for failed simulations

# ============================================================================
# PSO IMPLEMENTATION
# ============================================================================

class PSO:
    def __init__(self, bounds, n_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5):
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.n_dims = len(bounds)
        
        # Initialize particles
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (n_particles, self.n_dims))
        
        velocity_range = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range, velocity_range, (n_particles, self.n_dims))
        
        self.p_best_pos = self.positions.copy()
        self.p_best_scores = np.full(n_particles, -np.inf)
        self.g_best_pos = None
        self.g_best_score = -np.inf
        
        self.history = {'g_best': [], 'mean': [], 'positions': [], 'scores': []}
    
    def optimize(self, obj_func):
        start = timer()
        
        for iteration in range(self.max_iter):
            scores = np.array([obj_func(p) for p in self.positions])
            
            # Update personal bests
            improved = scores > self.p_best_scores
            self.p_best_scores[improved] = scores[improved]
            self.p_best_pos[improved] = self.positions[improved]
            
            # Update global best
            best_idx = np.argmax(scores)
            if scores[best_idx] > self.g_best_score:
                self.g_best_score = scores[best_idx]
                self.g_best_pos = self.positions[best_idx].copy()
            
            # Store history
            self.history['g_best'].append(self.g_best_score)
            self.history['mean'].append(np.mean(scores))
            self.history['positions'].append(self.positions.copy())
            self.history['scores'].append(scores.copy())
            
            print(f"Iter {iteration+1}/{self.max_iter}: "
                  f"Best={self.g_best_score:.4f}, Mean={np.mean(scores):.4f}, "
                  f"Time={(timer()-start)/(iteration+1):.1f}s/iter")
            
            # Update velocities and positions
            r1 = np.random.random((self.n_particles, self.n_dims))
            r2 = np.random.random((self.n_particles, self.n_dims))
            
            self.velocities = (self.w * self.velocities + 
                             self.c1 * r1 * (self.p_best_pos - self.positions) +
                             self.c2 * r2 * (self.g_best_pos - self.positions))
            
            self.positions = np.clip(self.positions + self.velocities,
                                    self.bounds[:, 0], self.bounds[:, 1])
        
        print(f"\nTotal time: {timer()-start:.1f}s")
        return self.g_best_pos, self.g_best_score
    
    def get_threshold_exceeding(self, threshold=0.0):
        """Get all parameters that exceeded threshold"""
        all_params = []
        for pos, scores in zip(self.history['positions'], self.history['scores']):
            for p, s in zip(pos, scores):
                if s > threshold:
                    all_params.append((*p, s))
        return np.array(all_params) if all_params else np.array([])
    
    def plot_results(self, threshold=0.0):
        """Generate convergence and parameter space plots"""
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Convergence plot
        ax1 = fig.add_subplot(131)
        iters = range(1, len(self.history['g_best']) + 1)
        ax1.plot(iters, self.history['g_best'], 'b-', lw=2, label='Global Best')
        ax1.plot(iters, self.history['mean'], 'r--', lw=1.5, label='Mean')
        ax1.axhline(y=threshold, color='k', ls=':', lw=1, label=f'Threshold={threshold}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness (Max Activity)')
        ax1.set_title('PSO Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 3D parameter space
        ax2 = fig.add_subplot(132, projection='3d')
        all_params = self.get_threshold_exceeding(threshold=-1e9)  # Get all
        
        if len(all_params) > 0:
            valid = all_params[all_params[:, 3] > threshold]
            invalid = all_params[all_params[:, 3] <= threshold]
            
            if len(invalid) > 0:
                ax2.scatter(invalid[:, 0], invalid[:, 1], invalid[:, 2],
                           c='lightgray', alpha=0.3, s=20, label='Below threshold')
            if len(valid) > 0:
                sc = ax2.scatter(valid[:, 0], valid[:, 1], valid[:, 2],
                               c=valid[:, 3], cmap='hot', alpha=0.7, s=50,
                               label='Above threshold')
                plt.colorbar(sc, ax=ax2, label='Fitness', shrink=0.5)
            
            ax2.scatter(self.g_best_pos[0], self.g_best_pos[1], self.g_best_pos[2],
                       c='blue', s=200, marker='*', edgecolors='black', lw=2,
                       label='Global Best')
        
        ax2.set_xlabel('frac_EE')
        ax2.set_ylabel('pulse_fre')
        ax2.set_zlabel('pulse_amp')
        ax2.set_title('3D Parameter Space')
        ax2.legend(fontsize=8)
        
        # 3. 2D projection (frac_EE vs pulse_fre)
        ax3 = fig.add_subplot(133)
        if len(all_params) > 0:
            valid = all_params[all_params[:, 3] > threshold]
            invalid = all_params[all_params[:, 3] <= threshold]
            
            if len(invalid) > 0:
                ax3.scatter(invalid[:, 0], invalid[:, 1], c='lightgray', alpha=0.3, s=20)
            if len(valid) > 0:
                ax3.scatter(valid[:, 0], valid[:, 1], c=valid[:, 3], 
                           cmap='hot', alpha=0.7, s=50)
            
            ax3.scatter(self.g_best_pos[0], self.g_best_pos[1],
                       c='blue', s=200, marker='*', edgecolors='black', lw=2)
        
        ax3.set_xlabel('frac_EE')
        ax3.set_ylabel('pulse_fre')
        ax3.set_title('frac_EE vs pulse_fre')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("PSO for Hexagonal Grid Wave Propagation Parameter Estimation")
    print("="*70)
    print("\nSearching parameter space:")
    print("  frac_EE:   (0.001, 0.5)")
    print("  pulse_fre: (0.001, 1.0)")
    print("  pulse_amp: (0.001, 5.0)")
    print("\nObjective: Maximize outer layer activity (last 12 oscillators, last 10k points)")
    print("="*70 + "\n")
    
    # Setup
    bounds = [(0.001, 0.5), (0.001, 1.0), (0.001, 5.0)]
    
    def objective(params):
        return simulate_hexgrid(params[0], params[1], params[2])
    
    # Run PSO
    pso = PSO(bounds=bounds, n_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5)
    best_params, best_fitness = pso.optimize(objective)
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Best parameters:")
    print(f"  frac_EE   = {best_params[0]:.6f}")
    print(f"  pulse_fre = {best_params[1]:.6f}")
    print(f"  pulse_amp = {best_params[2]:.6f}")
    print(f"  Fitness   = {best_fitness:.6f}")
    
    threshold_params = pso.get_threshold_exceeding(threshold=0.0)
    print(f"\nParameter combinations exceeding threshold (>0): {len(threshold_params)}")
    
    if len(threshold_params) > 0:
        sorted_params = threshold_params[threshold_params[:, 3].argsort()[::-1]]
        print("\nTop 5 combinations:")
        print(f"{'frac_EE':<12} {'pulse_fre':<12} {'pulse_amp':<12} {'fitness':<12}")
        print("-"*50)
        for params in sorted_params[:5]:
            print(f"{params[0]:<12.4f} {params[1]:<12.4f} {params[2]:<12.4f} {params[3]:<12.4f}")
    
    # Plot
    fig = pso.plot_results(threshold=0.0)
    fig.savefig('/mnt/user-data/outputs/pso_results.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to /mnt/user-data/outputs/pso_results.png")
    
    # Save detailed results
    with open('/mnt/user-data/outputs/pso_all_results.txt', 'w') as f:
        f.write("All Parameter Combinations Exceeding Threshold\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'frac_EE':<12} {'pulse_fre':<12} {'pulse_amp':<12} {'fitness':<12}\n")
        f.write("-"*50 + "\n")
        for params in sorted_params:
            f.write(f"{params[0]:<12.6f} {params[1]:<12.6f} {params[2]:<12.6f} {params[3]:<12.6f}\n")
    
    print(f"Detailed results saved to /mnt/user-data/outputs/pso_all_results.txt")
    
    return pso, best_params, best_fitness, threshold_params

if __name__ == "__main__":
    pso, best_params, best_fitness, threshold_params = main()
