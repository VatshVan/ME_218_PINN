import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
DIC_FILE = 'DIC_2D_Data.csv'
UTM_FILE = 'Specimen_RawData_1.csv'
CROSS_SECTIONAL_AREA = 13.99  # mm^2 (Physical dimensions of the specimen)
POISSONS_RATIO = 0.3
EPOCHS = 3000
BATCH_SIZE = 4096
CHECKPOINT_FILE = 'Weights_Al.pth'

# --- 2. DATA EXTRACTION ---
def load_elastic_dataset():
    # Load UTM
    utm_df = pd.read_csv(UTM_FILE, skiprows=[1]).apply(pd.to_numeric, errors='coerce').dropna()
    t_utm = utm_df['Time'].values
    load_utm = utm_df['Load'].values # Native Newtons
    
    # Isolate Elastic Region using amplitude thresholding (15% to 75% max load)
    # This bypasses the grip-seating noise and the plastic yielding entirely
    max_load = np.max(load_utm)
    idx_start = np.argmax(load_utm >= 0.15 * max_load)
    idx_end = np.argmax(load_utm >= 0.75 * max_load)
    t_start, t_end = t_utm[idx_start], t_utm[idx_end]
    
    print(f"Extracted Linear Elastic Window: {t_start:.2f}s to {t_end:.2f}s")
    
    # Load DIC
    dic_df = pd.read_csv(DIC_FILE)
    dic_df['t'] = dic_df['img_no'] / 10.0
    
    # Truncate and Subsample
    dic_df = dic_df[(dic_df['t'] >= t_start) & (dic_df['t'] <= t_end)]
    dic_df = dic_df.sample(frac=0.2, random_state=42) # Use 20% of the elastic zone
    
    t_dic = dic_df['t'].values
    sync_load = np.interp(t_dic, t_utm, load_utm)
    
    # Convert to Tensors
    t_t = torch.tensor(t_dic, dtype=torch.float32).view(-1, 1)
    x_t = torch.tensor(dic_df['x_pic_AVG'].values, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(dic_df['y_pic_AVG'].values, dtype=torch.float32).view(-1, 1)
    u_t = torch.tensor(dic_df['u_AVG'].values, dtype=torch.float32).view(-1, 1)
    v_t = torch.tensor(dic_df['v_AVG'].values, dtype=torch.float32).view(-1, 1)
    p_t = torch.tensor(sync_load, dtype=torch.float32).view(-1, 1)
    
    return x_t, y_t, t_t, u_t, v_t, p_t

# --- 3. HOMOGENEOUS PINN ARCHITECTURE ---
class VanillaPINN(nn.Module):
    def __init__(self, stats):
        super().__init__()
        # Z-score statistics for internal coordinate normalization
        self.register_buffer('x_mu', stats['x_mu']); self.register_buffer('x_std', stats['x_std'])
        self.register_buffer('y_mu', stats['y_mu']); self.register_buffer('y_std', stats['y_std'])
        self.register_buffer('t_mu', stats['t_mu']); self.register_buffer('t_std', stats['t_std'])
        
        # Standard MLP for (x, y, t) -> (u, v)
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
        
        # --- THE INVERSE PARAMETER ---
        # A single learnable scalar for Young's Modulus (in GPa). 
        # Initialized deliberately wrong (30 GPa) so the network is forced to find the ~70 GPa baseline.
        self.E_gpa = nn.Parameter(torch.tensor([30.0])) 

    def forward(self, x, y, t):
        # Normalize inputs for stable gradients
        x_n = (x - self.x_mu) / self.x_std
        y_n = (y - self.y_mu) / self.y_std
        t_n = (t - self.t_mu) / self.t_std
        
        uv = self.net(torch.cat([x_n, y_n, t_n], dim=1))
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        
        # Convert the scalar parameter to MPa for physics equations
        E_mpa = self.E_gpa * 1000.0 
        G_mpa = E_mpa / (2.0 * (1.0 + POISSONS_RATIO))
        
        return u, v, E_mpa, G_mpa

def get_grad(out, inp):
    # Standard autograd hook
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]

# --- 4. OPTIMIZATION PIPELINE ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_raw, y_raw, t_raw, u_emp, v_emp, p_emp = load_elastic_dataset()
    
    # Capture normalization statistics
    stats = {
        'x_mu': x_raw.mean(), 'x_std': x_raw.std(),
        'y_mu': y_raw.mean(), 'y_std': y_raw.std(),
        't_mu': t_raw.mean(), 't_std': t_raw.std(),
    }
    
    dataset = torch.utils.data.TensorDataset(x_raw, y_raw, t_raw, u_emp, v_emp, p_emp)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = VanillaPINN(stats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting Vanilla PINN Inverse Parameter Identification...")
    for epoch in range(EPOCHS):
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0
        epoch_stress_loss = 0.0
        
        for batch in dataloader:
            x_b, y_b, t_b, u_b, v_b, p_b = [t.to(device) for t in batch]
            x_b.requires_grad_(True)
            y_b.requires_grad_(True)
            
            optimizer.zero_grad()
            
            u_pred, v_pred, E_mpa, G_mpa = model(x_b, y_b, t_b)
            
            # --- LOSS 1: Kinematic Data (u, v) ---
            # Scaled up because displacements are tiny (mm)
            loss_data = torch.mean((u_pred - u_b)**2 + (v_pred - v_b)**2) * 1e6
            
            # --- LOSS 2: Quasi-Static PDE (Equilibrium) ---
            u_x = get_grad(u_pred, x_b); u_y = get_grad(u_pred, y_b)
            v_x = get_grad(v_pred, x_b); v_y = get_grad(v_pred, y_b)
            
            eps_xx = u_x
            eps_yy = v_y
            gamma_xy = u_y + v_x
            
            factor = E_mpa / (1.0 - POISSONS_RATIO**2)
            sigma_xx = factor * (eps_xx + POISSONS_RATIO * eps_yy)
            sigma_yy = factor * (eps_yy + POISSONS_RATIO * eps_xx)
            tau_xy = G_mpa * gamma_xy
            
            sig_xx_x = get_grad(sigma_xx, x_b)
            tau_xy_y = get_grad(tau_xy, y_b)
            tau_xy_x = get_grad(tau_xy, x_b)
            sig_yy_y = get_grad(sigma_yy, y_b)
            
            # Div(Sigma) = 0. Inertia (u_tt) is removed for static tensile testing.
            res_x = sig_xx_x + tau_xy_y
            res_y = tau_xy_x + sig_yy_y
            loss_pde = torch.mean(res_x**2 + res_y**2) 
            
            # --- LOSS 3: Global Force Boundary Condition ---
            # The average longitudinal stress must equal the applied UTM Load / Area
            target_stress = p_b / CROSS_SECTIONAL_AREA
            loss_stress = torch.mean((sigma_yy - target_stress)**2) 
            
            # Formulate Total Loss & Backpropagate
            loss = loss_data + loss_pde + loss_stress
            loss.backward()
            optimizer.step()
            
            epoch_data_loss += loss_data.item()
            epoch_pde_loss += loss_pde.item()
            epoch_stress_loss += loss_stress.item()
            
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            nb = len(dataloader)
            print(f"Epoch {epoch:4d} | Data: {epoch_data_loss/nb:.2f} | PDE: {epoch_pde_loss/nb:.2e} | Stress BC: {epoch_stress_loss/nb:.2e} || E_Predicted: {model.E_gpa.item():.2f} GPa")
        torch.save({
            'Epoch': epoch,
            'model_state_dict': model.state_dict(),
            'normalizer_mean': torch.tensor([stats['x_mu'], stats['y_mu'], stats['t_mu']]),
            'normalizer_std': torch.tensor([stats['x_std'], stats['y_std'], stats['t_std']])
        }, CHECKPOINT_FILE)