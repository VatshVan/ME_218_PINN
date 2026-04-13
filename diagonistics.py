import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from A_DHC import DHC_PINN

# --- CONFIGURATION (Sync with s.py) ---
CHECKPOINT_FILE = r"Data\Result\Lines_80\Batch_DHC_1.pth" 
X_LIMITS = (0, 30.30)  # Width w
Y_LIMITS = (0, 4.94)   # Thickness t/Height
QUERY_TIME = 150.0      # Evaluation point within the elastic window
DIR = "Diagnostics"

def analyze_equilibrium(directory=DIR):
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"Error: Checkpoint {CHECKPOINT_FILE} not found.")
        return

    checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')
    
    os.makedirs(directory, exist_ok=True)

    # 1. TELEMETRY EXTRACTION
    if 'history' not in checkpoint:
        print("Error: Telemetry 'history' not found in checkpoint. Ensure s.py is saving history.")
        return
        
    hist = checkpoint['history']
    epochs = hist['epoch']

    # 2. PLOT 1: MULTIPLIER STABILITY (Dual Path)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("Dual Optimization: Multiplier Evolution (Nash Equilibrium Search)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Multiplier Magnitude (Log Scale)")
    
    ax1.plot(epochs, hist['lagrangian'], label='Lagrangian Loss', color='purple', alpha=0.8, linewidth=2)
    ax1.plot(epochs, hist['l_pde'], label='λ_PDE', alpha=0.8)
    ax1.plot(epochs, hist['l_force'], label='λ_Force', alpha=0.8)
    ax1.plot(epochs, hist['l_diric'], label='λ_Dirichlet', alpha=0.8)
    ax1.plot(epochs, hist['l_neum'], label='λ_Neumann', alpha=0.8)
    ax1.plot(epochs, hist['l_smooth'], label='λ_Smooth', alpha=0.8)
    if 'l_strain' in hist:
        ax1.plot(epochs, hist['l_strain'], label='λ_Strain', linewidth=2, linestyle='--')
    
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()

    # 3. PLOT 2: RESIDUAL CONVERGENCE (Primal Path)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Raw Residual (Log Scale)")
    ax2.plot(epochs, hist['pde'], color='black', label='Raw PDE Loss', alpha=0.6, linestyle=':', linewidth=2)
    ax2.plot(epochs, hist['data'], color='orange', label='Raw Data Loss', alpha=0.6, linestyle=':', linewidth=2)
    ax2.set_yscale('log')
    ax2.legend(loc = 'upper center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "Equilibrium_Stability.png"), dpi=300)
    print("Telemetric stability plot saved: Equilibrium_Stability.png")
    plt.close()
    # 4. BOUNDARY CONDITION VERIFICATION
    # Extracting a slice along the Y-axis to check Dirichlet/Neumann adherence
    # This requires loading the model architecture (imported or redefined)
    # Since I cannot run the forward pass here, I recommend you run your 
    # generate_ultra_detailed_plots function already present in s.py.

def generate_equilibrium_spatial_maps(model_architecture, weights_file, x_limits, y_limits, query_time, grid_resolution=150, directory=DIR):
    """
    Revised with Point-Batching to prevent CUDA OOM.
    grid_resolution reduced from 300 to 150 (~4x fewer points).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_file, map_location=device)
    os.makedirs(directory, exist_ok=True)

    # 1. Model Restoration
    model = model_architecture().to(device)
    raw_state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace('surrogate.', ''): v for k, v in raw_state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    
    m, s = checkpoint['normalizer_mean'], checkpoint['normalizer_std']

    # 2. Quadrature Grid Generation
    x = np.linspace(x_limits[0], x_limits[1], grid_resolution)
    y = np.linspace(y_limits[0], y_limits[1], grid_resolution)
    mesh_x, mesh_y = np.meshgrid(x, y)
    
    # Flattened query tensors
    t_x_all = torch.tensor(mesh_x.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_y_all = torch.tensor(mesh_y.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_t_all = torch.full_like(t_x_all, float(query_time))

    # 3. Batch Processing Loop (Prevent VRAM Saturation)
    # Process 10,000 points at a time
    inference_batch_size = 10000 
    total_points = t_x_all.shape[0]
    
    pde_flat = []
    diric_flat = []
    neum_flat = []

    for i in range(0, total_points, inference_batch_size):
        end = min(i + inference_batch_size, total_points)
        
        # Slice batches and enable gradients locally
        b_x = t_x_all[i:end].clone().detach().requires_grad_(True)
        b_y = t_y_all[i:end].clone().detach().requires_grad_(True)
        b_t = t_t_all[i:end].clone().detach().requires_grad_(True)

        # Normalization
        b_x_n, b_y_n, b_t_n = (b_x-m['x'])/s['x'], (b_y-m['y'])/s['y'], (b_t-m['t'])/s['t']
        
        # Forward pass
        u_n, v_n, E, G = model(b_x_n, b_y_n, b_t_n)
        
        # Scalar Restoration Utilities
        def get_grad(out, inp, scale=1.0):
            return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0] * scale

        # Kinematics & Stress (Corrected for Normalization)
        eps_xx = get_grad(u_n, b_x, s['u']/s['x'])
        eps_yy = get_grad(v_n, b_y, s['v']/s['y'])
        gamma_xy = get_grad(u_n, b_y, s['u']/s['y']) + get_grad(v_n, b_x, s['v']/s['x'])
        
        nu = 0.35
        sigma_yy = (E / (1 - nu**2)) * (eps_yy + nu * eps_xx)
        tau_xy = G * gamma_xy
        sigma_xx = (E / (1 - nu**2)) * (eps_xx + nu * eps_yy)

        # PDE Residuals (Divergence of Stress)
        d_sig_xx_dx = get_grad(sigma_xx, b_x, 1.0/s['x'])
        d_tau_xy_dy = get_grad(tau_xy, b_y, 1.0/s['y'])
        d_tau_xy_dx = get_grad(tau_xy, b_x, 1.0/s['x'])
        d_sig_yy_dy = get_grad(sigma_yy, b_y, 1.0/s['y'])
        
        # Collect Local Results
        pde_flat.append(((d_sig_xx_dx + d_tau_xy_dy)**2 + (d_tau_xy_dx + d_sig_yy_dy)**2).detach())
        diric_flat.append(((u_n*s['u']+m['u'])**2 + (v_n*s['v']+m['v'])**2).detach())
        
        current_load = 1581.11 # Final load baseline
        traction_target = current_load / 149.682
        neum_flat.append(((sigma_yy - traction_target)**2).detach())

    # 4. Reconstruct Maps
    pde_map = torch.cat(pde_flat).cpu().numpy().reshape(grid_resolution, grid_resolution)
    diric_map = torch.cat(diric_flat).cpu().numpy().reshape(grid_resolution, grid_resolution)
    neum_map = torch.cat(neum_flat).cpu().numpy().reshape(grid_resolution, grid_resolution)

    # 5. Diagnostic Visualization
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # Magma for PDE, Reds for Dirichlet, Blues for Neumann
    im1 = axes[0].pcolormesh(mesh_x, mesh_y, pde_map, cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1e4))
    im2 = axes[1].pcolormesh(mesh_x, mesh_y, diric_map, cmap='Reds', norm=LogNorm(vmin=1e-6, vmax=1e0))
    im3 = axes[2].pcolormesh(mesh_x, mesh_y, neum_map, cmap='Blues', norm=LogNorm(vmin=1e-2, vmax=1e4))

    fig.colorbar(im1, ax=axes[0]); fig.colorbar(im2, ax=axes[1]); fig.colorbar(im3, ax=axes[2])
    axes[0].set_title("PDE Equilibrium Error")
    axes[1].set_title("Dirichlet Drift (Anchor)")
    axes[2].set_title("Neumann Traction Error")
    
    plt.tight_layout()
    plt.savefig(f"{directory}/Equilibrium_Topology_t{query_time}.png", dpi=300)
    plt.close()

def generate_residual_spatial_maps(model_architecture, weights_file, x_limits, y_limits, query_time, grid_resolution=300, directory=DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_file, map_location=device)
    os.makedirs(directory, exist_ok=True)

    # 1. Surrogate Restoration & State Synchronization
    model = model_architecture().to(device)
    raw_state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace('surrogate.', ''): v for k, v in raw_state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    
    # Extract Normalization Scalars (Crucial for Gradient Scaling)
    m, s = checkpoint['normalizer_mean'], checkpoint['normalizer_std']

    # 2. Quadrature Grid Generation
    x = np.linspace(x_limits[0], x_limits[1], grid_resolution)
    y = np.linspace(y_limits[0], y_limits[1], grid_resolution)
    mesh_x, mesh_y = np.meshgrid(x, y)
    
    t_x = torch.tensor(mesh_x.flatten(), dtype=torch.float32, requires_grad=True).view(-1, 1).to(device)
    t_y = torch.tensor(mesh_y.flatten(), dtype=torch.float32, requires_grad=True).view(-1, 1).to(device)
    t_t = torch.full_like(t_x, float(query_time), requires_grad=True)

    # Forward Pass in Normalized Space
    t_x_n, t_y_n, t_t_n = (t_x-m['x'])/s['x'], (t_y-m['y'])/s['y'], (t_t-m['t'])/s['t']
    u_n, v_n, E, G = model(t_x_n, t_y_n, t_t_n)
    
    # 3. High-Fidelity Physics Extraction
    def get_grad(out, inp, scale=1.0):
        g = torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]
        return g * scale

    # Kinematic Gradients (Restored to Physical Units)
    eps_xx = get_grad(u_n, t_x, s['u']/s['x'])
    eps_yy = get_grad(v_n, t_y, s['v']/s['y'])
    gamma_xy = get_grad(u_n, t_y, s['u']/s['y']) + get_grad(v_n, t_x, s['v']/s['x'])
    
    # Constitutive Integration (Plane Stress Assumption)
    nu = 0.35 # Effective Poisson's Ratio for 80% Gyroid PLA
    sigma_xx = (E / (1 - nu**2)) * (eps_xx + nu * eps_yy)
    sigma_yy = (E / (1 - nu**2)) * (eps_yy + nu * eps_xx)
    tau_xy = G * gamma_xy
    
    # 4. Residual Mapping Logic
    # PDE: Divergence of Stress - Inertia (Static regime inertia ~ 0)
    d_sig_xx_dx = get_grad(sigma_xx, t_x, 1.0/s['x'])
    d_tau_xy_dy = get_grad(tau_xy, t_y, 1.0/s['y'])
    d_tau_xy_dx = get_grad(tau_xy, t_x, 1.0/s['x'])
    d_sig_yy_dy = get_grad(sigma_yy, t_y, 1.0/s['y'])
    
    pde_res = (d_sig_xx_dx + d_tau_xy_dy)**2 + (d_tau_xy_dx + d_sig_yy_dy)**2
    pde_map = pde_res.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)

    # Dirichlet: Targeted at y=0 (Physical Grip Anchor)
    # Map shows where the specimen is 'slipping' from the anchor
    u_phys = (u_n * s['u']) + m['u']
    v_phys = (v_n * s['v']) + m['v']
    diric_res = (u_phys**2 + v_phys**2)
    diric_map = diric_res.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)

    # Neumann: Stress vs. Experimental Traction
    # Dynamic Load Lookup (Use the load specific to query_time)
    # Assumes p_target was synced in your UTM dataframe
    current_load = 1581.11 # Replace with interpolation: np.interp(query_time, t_utm, p_utm)
    traction_target = current_load / 149.682 
    neum_res = (sigma_yy - traction_target)**2
    neum_map = neum_res.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)

    # 5. Diagnostic Visualization
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # PDE Map: Internal Equilibrium Stability
    im1 = axes[0].pcolormesh(mesh_x, mesh_y, pde_map, cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1e4), shading='auto')
    axes[0].set_title("PDE Equilibrium Error ($|\\nabla \\cdot \\sigma|^2$)")
    fig.colorbar(im1, ax=axes[0])
    
    # Dirichlet Map: Domain Anchor Analysis (Should be 0 at y=0)
    im2 = axes[1].pcolormesh(mesh_x, mesh_y, diric_map, cmap='Reds', norm=LogNorm(vmin=1e-6, vmax=1e0), shading='auto')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title("Dirichlet Drift ($u^2 + v^2$)")
    fig.colorbar(im2, ax=axes[1])
    
    # Neumann Map: Stress-Load Synchronization (Should be 0 at y=100)
    im3 = axes[2].pcolormesh(mesh_x, mesh_y, neum_map, cmap='Blues', norm=LogNorm(vmin=1e-2, vmax=1e4), shading='auto')
    axes[2].axhline(y=100, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title("Neumann Traction Error ($(\sigma_{yy} - \sigma_{target})^2$)")
    fig.colorbar(im3, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Width (mm)"); ax.set_ylabel("Length (mm)")

    plt.tight_layout()
    plt.savefig(f"{directory}/Equilibrium_Topology_t{query_time}.png", dpi=300)
    plt.close()

if __name__ == "__main__":

    analyze_equilibrium(DIR)

    generate_equilibrium_spatial_maps(
        model_architecture=DHC_PINN, 
        weights_file=CHECKPOINT_FILE,
        x_limits=(0, 30.30), 
        y_limits=(0, 100), 
        query_time=150.0, # Mid-elastic window
        directory=DIR
    )
