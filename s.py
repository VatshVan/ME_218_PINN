import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.stats import linregress
from torch.utils.data import TensorDataset, DataLoader
torch.set_float32_matmul_precision('high')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

DIC_FILE = 'DIC_2D_Data.csv'
UTM_FILE = 'Specimen_RawData_1.csv'
CHECKPOINT_FILE = 'Batch_1.pth'

CROSS_SECTIONAL_AREA = 13.99 * 1.0 
POISSONS_RATIO = 0.3
MATERIAL_DENSITY = 2.7e-9 
DIC_FRAME_RATE = 10.0 
E_REFERENCE = 70e3

# Optimization Hyperparameters
EPOCHS = 2000
LR_THETA = 1e-3
LR_LAMBDA = 1e-2       # Learning rate for the Lagrangian Multipliers
BATCH_SIZE = 8192
LAMBDA_DATA = 1.0      # Anchor primal loss (static)

def smooth_displacement_field(df: pd.DataFrame, cols=('u_AVG', 'v_AVG'), window: int = 11, polyorder: int = 3) -> pd.DataFrame:
    df_out = df.copy()
    for frame_id, group in df_out.groupby('img_no'):
        idx = group.index
        for col in cols:
            vals = group[col].values
            win = min(window, len(vals) if len(vals) % 2 == 1 else len(vals) - 1)
            win = max(win, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
            if len(vals) >= win:
                df_out.loc[idx, col] = savgol_filter(vals, window_length=win, polyorder=polyorder)
    return df_out

def isolate_hookean_regime(time_array, load_array, window_size=50, r_squared_threshold=0.999):
    peak_idx = np.argmax(load_array)
    t_trunc = time_array[:peak_idx]
    l_trunc = load_array[:peak_idx]
    
    r2_scores = []
    
    for i in range(len(l_trunc) - window_size):
        t_window = t_trunc[i:i + window_size]
        l_window = l_trunc[i:i + window_size]
        
        _, _, r_value, _, _ = linregress(t_window, l_window)
        r2_scores.append(r_value**2)
        
    r2_array = np.array(r2_scores)
    linear_blocks = np.where(r2_array >= r_squared_threshold)[0]
    
    if len(linear_blocks) == 0:
        raise ValueError("Optimal linear manifold not detected within defined thresholds.")
        
    start_idx = linear_blocks[0]
    end_idx = linear_blocks[-1] + window_size
    
    print(f"Identified Linear Regime: Start Time = {t_trunc[start_idx]:.4f}s, End Time = {t_trunc[end_idx]:.4f}s, R² = {r2_array[linear_blocks].max():.4f}")
    print(f"Load at Start: {l_trunc[start_idx]:.2f} N, Load at End: {l_trunc[end_idx]:.2f} N")
    return t_trunc[start_idx], t_trunc[end_idx]

class Normalizer:
    def __init__(self):
        self.mean: dict[str, float] = {}
        self.std:  dict[str, float] = {}

    def fit_transform(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        m = tensor.mean().item()
        s = tensor.std().item()
        s = s if s > 1e-8 else 1.0
        self.mean[name] = m
        self.std[name]  = s
        return (tensor - m) / s

    def inverse(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor * self.std[name] + self.mean[name]

def process_synchronized_tensors(dic_path: str, utm_path: str, checkpoint_norm: Normalizer = None, subsample_ratio: float = 0.2):
    utm_df = pd.read_csv(utm_path, skiprows=[1])
    utm_df = utm_df.apply(pd.to_numeric, errors='coerce').dropna()

    utm_time = utm_df['Time'].values
    utm_load = utm_df['Load'].values # Assuming Corrected to Newtons natively
    filt_load = savgol_filter(utm_load, window_length=51, polyorder=3)

    dt = np.mean(np.diff(utm_time))
    load_dot = np.gradient(filt_load, dt)
    load_ddot = np.gradient(load_dot, dt)

    # --- EXECUTE TEMPORAL BANDPASS FILTER ---
    t_start, t_end = isolate_hookean_regime(utm_time, filt_load)

    dic_df = pd.read_csv(dic_path)
    dic_df['Absolute_Time'] = dic_df['img_no'] / DIC_FRAME_RATE

    # Chronologically truncate the spatial tracking data
    original_frames = dic_df['img_no'].nunique()
    dic_df = dic_df[(dic_df['Absolute_Time'] >= t_start) & (dic_df['Absolute_Time'] <= t_end)]
    isolated_frames = dic_df['img_no'].nunique()
    print(f"Dataset Truncated: Reduced from {original_frames} to {isolated_frames} temporal frames.")

    # Execute spatial smoothing on the isolated block
    dic_df = smooth_displacement_field(dic_df)

    # --- EXECUTE SPATIAL SUBSAMPLING ---
    if subsample_ratio < 1.0:
        original_size = len(dic_df)
        dic_df = dic_df.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
        print(f"Spatial Subsampling: Reduced tensor dimension from {original_size} to {len(dic_df)} nodes.")

    dic_time = dic_df['Absolute_Time'].values
    sync_load = np.interp(dic_time, utm_time, filt_load)
    sync_load_dot = np.interp(dic_time, utm_time, load_dot)
    sync_load_ddot = np.interp(dic_time, utm_time, load_ddot)

    t_raw = torch.tensor(dic_time, dtype=torch.float32).view(-1, 1)
    x_raw = torch.tensor(dic_df['x_pic_AVG'].values, dtype=torch.float32).view(-1, 1)
    y_raw = torch.tensor(dic_df['y_pic_AVG'].values, dtype=torch.float32).view(-1, 1)
    u_raw = torch.tensor(dic_df['u_AVG'].values, dtype=torch.float32).view(-1, 1)
    v_raw = torch.tensor(dic_df['v_AVG'].values, dtype=torch.float32).view(-1, 1)

    p_tensor = torch.tensor(sync_load, dtype=torch.float32).view(-1, 1)
    p_dot_tensor = torch.tensor(sync_load_dot, dtype=torch.float32).view(-1, 1)
    p_ddot_tensor = torch.tensor(sync_load_ddot, dtype=torch.float32).view(-1, 1)

    norm = checkpoint_norm if checkpoint_norm is not None else Normalizer()

    if checkpoint_norm is not None:
        x_n = (x_raw - norm.mean['x']) / norm.std['x']
        y_n = (y_raw - norm.mean['y']) / norm.std['y']
        t_n = (t_raw - norm.mean['t']) / norm.std['t']
        u_n = (u_raw - norm.mean['u']) / norm.std['u']
        v_n = (v_raw - norm.mean['v']) / norm.std['v']
    else:
        x_n = norm.fit_transform(x_raw, 'x')
        y_n = norm.fit_transform(y_raw, 'y')
        t_n = norm.fit_transform(t_raw, 't')
        u_n = norm.fit_transform(u_raw, 'u')
        v_n = norm.fit_transform(v_raw, 'v')

    y_min_norm = ((y_raw.min().item() - norm.mean['y']) / norm.std['y'])
    y_max_norm = ((y_raw.max().item() - norm.mean['y']) / norm.std['y'])
    eps = 0.05 

    bottom_mask = (y_n < y_min_norm + eps).squeeze()
    top_mask = (y_n > y_max_norm - eps).squeeze()

    return x_n, y_n, t_n, u_n, v_n, p_tensor, p_dot_tensor, p_ddot_tensor, bottom_mask, top_mask, norm

class FourierFeatureEncoding(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class OptimizedBifurcatedPINN(nn.Module):
    def __init__(self, fourier_features: int = 64):
        super().__init__()
        # Separate encoders: kinematics uses higher freq, material uses lower
        self.encoder_kin = FourierFeatureEncoding(3, fourier_features, scale=1.0)
        self.encoder_mat = FourierFeatureEncoding(3, fourier_features // 2, scale=0.25)

        feat_kin = fourier_features * 2
        feat_mat = fourier_features  # half features, so still *2 from sin/cos

        # Kinematic branch (unchanged — it's working)
        self.kin_trunk = nn.Sequential(
            nn.Linear(feat_kin, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.kinematic_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2),
        )

        # Dedicated material branch — fully separated
        self.mat_trunk = nn.Sequential(
            nn.Linear(feat_mat, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
        )
        self.constitutive_head = nn.Sequential(
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, y, t):
        coords = torch.cat([x, y, t], dim=1)

        # Kinematic path
        kin_enc = self.encoder_kin(coords)
        kin_lat = self.kin_trunk(kin_enc)
        kinematics = self.kinematic_head(kin_lat)
        u = kinematics[:, 0:1]
        v = kinematics[:, 1:2]

        # Material path — use only spatial coords (x, y), not t
        # E should not vary in time for a solid specimen
        xy_only = torch.cat([x, y, torch.zeros_like(t)], dim=1)
        mat_enc = self.encoder_mat(xy_only)
        mat_lat = self.mat_trunk(mat_enc)
        E_raw = self.constitutive_head(mat_lat)
        E = (torch.nn.functional.softplus(E_raw) + 0.5) * E_REFERENCE
        G = E / (2.0 * (1.0 + POISSONS_RATIO))

        return u, v, E, G

class AdaptiveLagrangianPINN(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.surrogate = base_model
        
        # # Initialized to match previous stable manual tuning equivalents in log space
        # self.log_l_pde = nn.Parameter(torch.tensor([-13.8]))    # ~ 1e-6
        # self.log_l_force = nn.Parameter(torch.tensor([-8.5]))   # ~ 2e-4
        # self.log_l_diric = nn.Parameter(torch.tensor([0.0]))    # ~ 1.0
        # self.log_l_neum = nn.Parameter(torch.tensor([-2.3]))    # ~ 1e-1
        # FIX 3: Re-initialize log multipliers with better priors

        # PDE residuals ~1e4, data ~1.0, so PDE needs λ ~ 1e-4 minimum
        self.log_l_pde = nn.Parameter(torch.tensor([-9.2]))    # ~1e-4
        self.log_l_force = nn.Parameter(torch.tensor([-7.0]))  # ~9e-4
        self.log_l_diric = nn.Parameter(torch.tensor([0.0]))   # ~1.0
        self.log_l_neum = nn.Parameter(torch.tensor([-1.0]))   # ~0.37

    def forward(self, x, y, t):
        return self.surrogate(x, y, t)

    def get_multipliers(self):
        return torch.exp(self.log_l_pde), torch.exp(self.log_l_force), torch.exp(self.log_l_diric), torch.exp(self.log_l_neum)

# --- REFACTORED: Returns Raw Unscaled Residuals ---
def compute_raw_residuals(model, x, y, t, u_emp, v_emp, p_emp, p_dot_emp, p_ddot_emp, bottom_mask, top_mask, norm: Normalizer):
    u_pred, v_pred, E_pred, G_pred = model(x, y, t)

    loss_data = torch.mean((u_pred - u_emp)**2 + (v_pred - v_emp)**2)

    def grad(out_tensor, inp_tensor):
        g = torch.autograd.grad(out_tensor, inp_tensor, torch.ones_like(out_tensor), create_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(inp_tensor)

    # --- DIMENSIONAL RESTORATION PROTOCOL ---
    std_x = norm.std['x']; std_y = norm.std['y']; std_t = norm.std['t']
    std_u = norm.std['u']; std_v = norm.std['v']

    u_x_norm = grad(u_pred, x); u_y_norm = grad(u_pred, y)
    v_x_norm = grad(v_pred, x); v_y_norm = grad(v_pred, y)

    eps_xx = u_x_norm * (std_u / std_x)
    eps_yy = v_y_norm * (std_v / std_y)
    gamma_xy = (u_y_norm * (std_u / std_y)) + (v_x_norm * (std_v / std_x))

    factor = E_pred / (1.0 - POISSONS_RATIO**2)
    sigma_xx = factor * (eps_xx + POISSONS_RATIO * eps_yy)
    sigma_yy = factor * (eps_yy + POISSONS_RATIO * eps_xx)
    tau_xy = G_pred * gamma_xy

    sigma_xx_x = grad(sigma_xx, x) / std_x
    tau_xy_y   = grad(tau_xy, y) / std_y
    tau_xy_x   = grad(tau_xy, x) / std_x
    sigma_yy_y = grad(sigma_yy, y) / std_y

    u_t_norm = grad(u_pred, t)
    u_tt_norm = grad(u_t_norm, t)
    v_t_norm = grad(v_pred, t)
    v_tt_norm = grad(v_t_norm, t)

    u_tt_phys = u_tt_norm * (std_u / (std_t**2))
    v_tt_phys = v_tt_norm * (std_v / (std_t**2))

    inertia_x = MATERIAL_DENSITY * u_tt_phys
    inertia_y = MATERIAL_DENSITY * v_tt_phys

    res_x = (sigma_xx_x + tau_xy_y) - inertia_x
    res_y = (tau_xy_x + sigma_yy_y) - inertia_y
    loss_pde = torch.mean(res_x**2 + res_y**2)

    sigma_yy_t = grad(sigma_yy, t) / std_t
    sigma_yy_tt = grad(sigma_yy_t, t) / std_t

    force_dot_local = sigma_yy_t * CROSS_SECTIONAL_AREA
    force_ddot_local = sigma_yy_tt * CROSS_SECTIONAL_AREA

    loss_bc_rate = torch.mean((force_dot_local - p_dot_emp)**2)
    loss_bc_accel = torch.mean((force_ddot_local - p_ddot_emp)**2)
    loss_force_bc = loss_bc_rate + loss_bc_accel

    if bottom_mask.any():
        u_bottom_phys = (u_pred[bottom_mask] * std_u) + norm.mean['u']
        v_bottom_phys = (v_pred[bottom_mask] * std_v) + norm.mean['v']
        loss_dirichlet = torch.mean(u_bottom_phys**2 + v_bottom_phys**2)
    else:
        loss_dirichlet = torch.tensor(0.0, device=x.device)

    # CORRECT — compute grad over full tensor, then mask
    if top_mask.any():
        sigma_yy_top = sigma_yy[top_mask]
        traction_target = (p_emp[top_mask] / (CROSS_SECTIONAL_AREA + 1e-12))
        loss_neumann = torch.mean((sigma_yy_top - traction_target)**2)
    # if top_mask.any():
    #   sigma_yy_top = sigma_yy[top_mask]
    #   traction_target = (p_emp[top_mask] / (CROSS_SECTIONAL_AREA + 1e-12))
    #   loss_neumann = torch.mean((sigma_yy_top - traction_target)**2)
    else:
        loss_neumann = torch.tensor(0.0, device=x.device)

    # Note: Returning Raw, Unscaled Losses
    return loss_data, loss_pde, loss_force_bc, loss_dirichlet, loss_neumann

def generate_spatial_tensor_maps(model_architecture, weights_file, x_limits, y_limits, query_time, grid_resolution=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Inference on Device: {device.type.upper()}")
    
    # 1. State Dictionary & Normalization Extraction
    checkpoint = torch.load(weights_file, map_location=device)
    
    # --- CRITICAL FIX: Strip the wrapper prefix from the checkpoint keys ---
    raw_state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k.replace('surrogate.', ''): v for k, v in raw_state_dict.items()}
    
    surrogate_model = model_architecture().to(device)
    surrogate_model.load_state_dict(clean_state_dict, strict=False)
    surrogate_model.eval()
    
    norm_mean = checkpoint['normalizer_mean']
    norm_std = checkpoint['normalizer_std']

    # 2. Spatial Grid Generation
    mesh_x, mesh_y = np.mgrid[x_limits[0]:x_limits[1]:complex(0, grid_resolution), 
                              y_limits[0]:y_limits[1]:complex(0, grid_resolution)]
    
    vector_x = mesh_x.flatten()
    vector_y = mesh_y.flatten()
    vector_t = np.full_like(vector_x, float(query_time))

    t_x = torch.tensor(vector_x, dtype=torch.float32).view(-1, 1).to(device)
    t_y = torch.tensor(vector_y, dtype=torch.float32).view(-1, 1).to(device)
    t_t = torch.tensor(vector_t, dtype=torch.float32).view(-1, 1).to(device)

    # 3. Latent Space Projection
    t_x_norm = (t_x - norm_mean['x']) / norm_std['x']
    t_y_norm = (t_y - norm_mean['y']) / norm_std['y']
    t_t_norm = (t_t - norm_mean['t']) / norm_std['t']

    # 4. Forward Pass
    with torch.no_grad():
        u_tensor, v_tensor, E_tensor, G_tensor = surrogate_model(t_x_norm, t_y_norm, t_t_norm)

    u_phys = (u_tensor * norm_std['u']) + norm_mean['u']
    v_phys = (v_tensor * norm_std['v']) + norm_mean['v']

    # 5. Output Projection to CPU
    matrix_E = E_tensor.cpu().numpy().reshape(grid_resolution, grid_resolution)
    matrix_G = G_tensor.cpu().numpy().reshape(grid_resolution, grid_resolution)
    matrix_u = u_phys.cpu().numpy().reshape(grid_resolution, grid_resolution)
    matrix_v = v_phys.cpu().numpy().reshape(grid_resolution, grid_resolution)

    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    contour_e = axes[0].contourf(mesh_x, mesh_y, matrix_E, levels=100, cmap='viridis')
    fig.colorbar(contour_e, ax=axes[0])
    axes[0].set_title(f'Spatially-Varying Elastic Modulus E(x,y) | t={query_time}')
    
    contour_g = axes[1].contourf(mesh_x, mesh_y, matrix_G, levels=100, cmap='plasma')
    fig.colorbar(contour_g, ax=axes[1])
    axes[1].set_title(f'Spatially-Varying Shear Modulus G(x,y) | t={query_time}')
    
    plt.tight_layout()
    plt.savefig(f'Constitutive_Field_t{query_time}.png', dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    contour_u = axes[0].contourf(mesh_x, mesh_y, matrix_u, levels=100, cmap='coolwarm')
    fig.colorbar(contour_u, ax=axes[0])
    axes[0].set_title(f'Predicted Displacement u(x,y) | t={query_time}')
    contour_v = axes[1].contourf(mesh_x, mesh_y, matrix_v, levels=100, cmap='coolwarm')
    fig.colorbar(contour_v, ax=axes[1])
    axes[1].set_title(f'Predicted Displacement v(x,y) | t={query_time}')
    plt.tight_layout()
    plt.savefig(f'Displacement_Field_t{query_time}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    t = time.time()
    t_1 = time.localtime(t)

    print(f"Execution Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', t_1)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device.type.upper()}")
    
    if not (os.path.exists(DIC_FILE) and os.path.exists(UTM_FILE)):
        print(f"ERROR: Required data files '{DIC_FILE}' or '{UTM_FILE}' not found.")
    else:
        if os.path.exists(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
            saved_norm = Normalizer()
            saved_norm.mean = checkpoint['normalizer_mean']
            saved_norm.std = checkpoint['normalizer_std']
            X, Y, T, U_exp, V_exp, P_emp, P_dot, P_ddot, bottom_mask, top_mask, norm = process_synchronized_tensors(DIC_FILE, UTM_FILE, saved_norm, subsample_ratio=0.5)
        else:
            X, Y, T, U_exp, V_exp, P_emp, P_dot, P_ddot, bottom_mask, top_mask, norm = process_synchronized_tensors(DIC_FILE, UTM_FILE, subsample_ratio=0.5)        
            
        dataset = TensorDataset(X, Y, T, U_exp, V_exp, P_emp, P_dot, P_ddot, bottom_mask, top_mask)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        # raw_state_dict = checkpoint['model_state_dict']
        # clean_state_dict = {k.replace('surrogate.', ''): v for k, v in raw_state_dict.items()}
    
        # --- Instantiate Models and Dual Optimizers ---
        base_model = OptimizedBifurcatedPINN()
        model_wrapper = AdaptiveLagrangianPINN(base_model).to(device)

        optimizer_theta = torch.optim.Adam(model_wrapper.surrogate.parameters(), lr=LR_THETA)
        
        # Maximize=True explicitly performs Gradient Ascent for the Lagrangian Multipliers
        optimizer_lambda = torch.optim.Adam([
            model_wrapper.log_l_pde,
            model_wrapper.log_l_force,
            model_wrapper.log_l_diric,
            model_wrapper.log_l_neum
        ], lr=LR_LAMBDA, maximize=True)

        start_epoch = 0

        # --- Robust Legacy Checkpoint Loader ---
        if os.path.exists(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
            if 'model_state_dict' in checkpoint:
                # Load weights strictly into surrogate to accommodate wrapper
                try:
                    model_wrapper.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError:
                    # Fallback for legacy checkpoints that didn't have the wrapper struct
                    model_wrapper.surrogate.load_state_dict(checkpoint['model_state_dict'], strict=False)

                if 'optimizer_theta_state_dict' in checkpoint:
                    optimizer_theta.load_state_dict(checkpoint['optimizer_theta_state_dict'])
                    optimizer_lambda.load_state_dict(checkpoint['optimizer_lambda_state_dict'])
                elif 'optimizer_state_dict' in checkpoint:
                    optimizer_theta.load_state_dict(checkpoint['optimizer_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}.")
            else:
                model_wrapper.surrogate.load_state_dict(checkpoint, strict=False)
                print("Legacy checkpoint loaded into base surrogate; restarting epoch counter.")
        else:
            print("No checkpoint found. Initialising from scratch.")

        history_log = {
            'epoch': [],
            'lagrangian': [],
            'data': [],
            'pde': [],
            'l_pde': [],
            'l_force': [],
            'l_diric': [],
            'l_neum': []
        }

        if os.path.exists(CHECKPOINT_FILE):
            if 'history' in checkpoint:
                history_log = checkpoint['history']
        
        for epoch in range(start_epoch, EPOCHS):
            model_wrapper.train()
            
            epoch_lagrangian = 0.0
            epoch_data = 0.0
            epoch_pde = 0.0

            for batch_idx, batch_data in enumerate(dataloader):
                x_b, y_b, t_b, u_b, v_b, p_b, p_dot_b, p_ddot_b, b_mask, t_mask = [
                    tensor.to(device) for tensor in batch_data
                ]

                x_b.requires_grad_(True)
                y_b.requires_grad_(True)
                t_b.requires_grad_(True)

                optimizer_theta.zero_grad()
                optimizer_lambda.zero_grad()

                # Get Unscaled Residuals
                l_data, l_pde, l_force, l_diric, l_neum = compute_raw_residuals(
                    model_wrapper.surrogate, x_b, y_b, t_b, u_b, v_b, p_b, p_dot_b, p_ddot_b, b_mask, t_mask, norm
                )

                # Get Current Adaptive Lambda Weights
                cur_l_pde, cur_l_force, cur_l_diric, cur_l_neum = model_wrapper.get_multipliers()

                # Formulate the Adaptive Lagrangian Objective
                lagrangian_loss = (LAMBDA_DATA * l_data) + \
                                  (cur_l_pde * l_pde) + \
                                  (cur_l_force * l_force) + \
                                  (cur_l_diric * l_diric) + \
                                  (cur_l_neum * l_neum)

                # Backpropagate through both Neural Weights and Multipliers
                lagrangian_loss.backward()
                
                optimizer_theta.step()
                # optimizer_lambda.step()
                if batch_idx % 5 == 0:
                    optimizer_lambda.step()

                epoch_lagrangian += lagrangian_loss.item()
                epoch_data += l_data.item()
                epoch_pde += l_pde.item()

            num_batches = len(dataloader)
            avg_lagrangian = epoch_lagrangian / num_batches
            avg_data = epoch_data / num_batches
            avg_pde = epoch_pde / num_batches

            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                # Capture Lambdas for logging
                with torch.no_grad():
                    log_pde, log_force, log_diric, log_neum = model_wrapper.get_multipliers()
                
                history_log['epoch'].append(epoch)
                history_log['lagrangian'].append(avg_lagrangian)
                history_log['data'].append(avg_data)
                history_log['pde'].append(avg_pde)
                history_log['l_pde'].append(log_pde.item())
                history_log['l_force'].append(log_force.item())
                history_log['l_diric'].append(log_diric.item())
                history_log['l_neum'].append(log_neum.item())

                print(
                    f"Epoch [{epoch:>4}/{EPOCHS}] | "
                    f"Lagrangian: {avg_lagrangian:.4e} | "
                    f"Data (Raw): {avg_data:.4e} | "
                    f"PDE (Raw): {avg_pde:.4e} || "
                    f"λ_PDE: {log_pde.item():.2e} | "
                    f"λ_Force: {log_force.item():.2e} | "
                    f"λ_Diric: {log_diric.item():.2e} | "
                    f"λ_Neum: {log_neum.item():.2e}"
                    f" | Time Elapsed: {time.time() - t:.2f}s"
                )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_wrapper.state_dict(),
                'optimizer_theta_state_dict': optimizer_theta.state_dict(),
                'optimizer_lambda_state_dict': optimizer_lambda.state_dict(),
                'loss': avg_lagrangian,
                'normalizer_mean': norm.mean,
                'normalizer_std': norm.std,
                'log_l_pde': model_wrapper.log_l_pde.detach().cpu(),
                'log_l_force': model_wrapper.log_l_force.detach().cpu(),
                'log_l_diric': model_wrapper.log_l_diric.detach().cpu(),
                'log_l_neum': model_wrapper.log_l_neum.detach().cpu(),
                'history': history_log
            }, CHECKPOINT_FILE)

        print("Training complete. Checkpoint saved.")

        generate_spatial_tensor_maps(
        model_architecture=OptimizedBifurcatedPINN, 
        weights_file="Batch_1.pth",
        x_limits=(0, 13.9), 
        y_limits=(0, 74), 
        query_time=10
    )