"""
A_DHC.py — Direct Hybrid Collocation PINN (IE-PINN)
=====================================================
Architecture: Inverse Elasticity PINN using the Direct Hybrid Collocation method.
Reference   : Raissi et al. (2019), Haghighat et al. (2021)

Core paradigm change vs. RBF pre-interpolation
-----------------------------------------------
RBF treats the 8 DIC blocks as isolated peaks and extrapolates gradients freely,
producing non-physical displacements (~9,000 mm) and strains (~26,000%).

DHC instead uses the Neural Network itself as the interpolator:
  • Data term  — 8 DIC blocks serve as sparse Dirichlet anchors (exact fit).
  • Physics term — 10 000+ Sobol-quasi-random collocation points enforce the
                   2-D Cauchy Momentum PDE across the full specimen domain.
  • The PDE prevents "hallucinated" waves: a 26 000 % strain gradient would
    produce a massive stress divergence that the optimizer immediately penalizes.

CSV coordinate note
-------------------
  The processed CSV uses columns (x_phys, y_phys, u_train, v_train).
  Auto-scaling maps these to physical specimen mm dimensions.
  Override COORD_SCALE_X / COORD_SCALE_Y / DISP_SCALE if auto-detect is wrong.
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.stats import linregress
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

torch.set_float32_matmul_precision('high')

# =============================================================================
#  FILE PATHS
# =============================================================================
DIC_FILE        = r"Data\Gyroid_60\Gyroid_60_DIC_Data_Processed.csv"
UTM_FILE        = r"Data\60_Gyroid.CSV"
CHECKPOINT_FILE = r"Data\Result\Gyroid_60\Batch_DHC_1.pth"
OUTPUT_DIR      = r"Data\Result\Gyroid_60\Plots_DHC_1"

# =============================================================================
#  MATERIAL CONSTANTS  (80% Gyroid)
#  Density → 0 invokes the quasi-static form of the PDE (no inertia term).
# =============================================================================
#   80 % Gyroid  : ρ=1000 kg/m³, ν=0.35, t=4.94 mm, w=30.30 mm, E_ref=1.8 GPa
#   60 % Gyroid  : ρ= 860 kg/m³, ν=0.36, t=4.98 mm, w=30.04 mm, E_ref=1.3 GPa
#   80 % Lines   : ρ=1050 kg/m³, ν=0.34, t=5.00 mm, w=29.745mm, E_ref=2.25 GPa

CROSS_SECTIONAL_AREA = 30.04 * 4.98    # mm²
POISSONS_RATIO       = 0.36
MATERIAL_DENSITY     = 0.860e-9        # t/mm³ — set 0.860e-9 to include inertia term
DIC_FRAME_RATE       = 10.0            # Hz
E_REFERENCE          = 1.3e3           # MPa (1.3 GPa) — softplus lower-bound anchor
X_PHYS_MAX           = 30.04           # mm (full specimen width)
Y_PHYS_MAX           = 100.0           # mm (full specimen gauge length)

# Calibration — None means auto-detect from data range.
# x_mm = x_csv * COORD_SCALE_X  (similarly for y, displacements)
COORD_SCALE_X   = None   # float or None
COORD_SCALE_Y   = None   # float or None
DISP_SCALE      = 1.0   # float or None  (target: displacements in mm)

# =============================================================================
#  TRAINING HYPERPARAMETERS
# =============================================================================
STRIDE_FACTOR         = 15       # Use every Nth DIC frame
SUBSAMPLE_RATIO       = 1.0     # Fraction of rows kept after temporal isolation

EPOCHS                = 10000
LR_THETA              = 1e-3     # Network weights
LR_LAMBDA             = 1e-3     # Lagrangian multipliers
BATCH_SIZE_DATA       = 16_384   # DIC anchor mini-batch size

# DHC collocation (PDE-only points)
N_COLLOC              = 10_000   # Total Sobol collocation points
BATCH_SIZE_COLLOC     = 16_384   # PDE mini-batch size
RESAMPLE_COLLOC_EVERY = 100      # Re-draw collocation cloud every N epochs

# Static anchor weights
LAMBDA_DATA           = 1.00
EPS_STRAIN            = 1e-3     # Strain noise-floor threshold (dimensionless)

# Fourier feature dimensions
FOURIER_KIN           = 64
FOURIER_MAT           = 32

# Linear regime detection (Hookean isolation)
R2_THRESHOLD          = 0.999
HOOKEAN_WINDOW        = 5


# =============================================================================
#  UTILITY CLASSES & FUNCTIONS
# =============================================================================

class Normalizer:
    """Zero-mean / unit-variance normalizer with named channels."""

    def __init__(self):
        self.mean: dict[str, float] = {}
        self.std:  dict[str, float] = {}

    def fit_transform(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        m = float(tensor.mean())
        s = float(tensor.std())
        s = s if s > 1e-8 else 1.0
        self.mean[name] = m
        self.std[name]  = s
        return (tensor - m) / s

    def transform(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return (tensor - self.mean[name]) / self.std[name]

    def inverse(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor * self.std[name] + self.mean[name]


def smooth_displacement_field(
    df: pd.DataFrame,
    cols: tuple = ('u_train', 'v_train'),
    window: int = 11,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Apply Savitzky-Golay temporal smoothing per DIC block."""
    df_out = df.copy()
    for _, group in df_out.groupby('img_no'):
        idx = group.index
        for col in cols:
            vals = group[col].values
            win = min(window, len(vals))
            if win % 2 == 0:
                win -= 1
            win = max(win, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
            if len(vals) >= win:
                df_out.loc[idx, col] = savgol_filter(vals, window_length=win, polyorder=polyorder)
    return df_out


def isolate_hookean_regime(
    time_array: np.ndarray,
    load_array: np.ndarray,
    window_size: int = HOOKEAN_WINDOW,
    r2_threshold: float = R2_THRESHOLD,
    save_path: str | None = None,
) -> tuple[float, float]:
    """
    Identify the linear-elastic (Hookean) regime via rolling R² analysis.
    Returns (t_start, t_end) bounding the highest-linearity block of frames.
    """
    peak_idx = int(np.argmax(load_array))
    t_trunc  = time_array[:peak_idx]
    l_trunc  = load_array[:peak_idx]

    r2_scores = np.array([
        linregress(t_trunc[i:i + window_size], l_trunc[i:i + window_size]).rvalue ** 2
        for i in range(len(l_trunc) - window_size)
    ])

    linear_blocks = np.where(r2_scores >= r2_threshold)[0]
    if len(linear_blocks) == 0:
        raise ValueError(
            f"No linear manifold found with R² ≥ {r2_threshold:.3f}. "
            "Lower R2_THRESHOLD or increase HOOKEAN_WINDOW."
        )

    # Trim window edges to stay well inside the linear regime
    margin = window_size
    start_idx = linear_blocks[0]  + margin
    end_idx   = linear_blocks[-1] - margin

    if start_idx >= end_idx:
        warnings.warn(
            "Linear regime is shorter than 2 × HOOKEAN_WINDOW. "
            "Using raw detected bounds; check data quality.",
            RuntimeWarning,
        )
        start_idx = linear_blocks[0]
        end_idx   = linear_blocks[-1]

    t_s, t_e = float(t_trunc[start_idx]), float(t_trunc[end_idx])
    print(
        f"[Hookean] t ∈ [{t_s:.4f}, {t_e:.4f}] s | "
        f"Load ∈ [{l_trunc[start_idx]:.2f}, {l_trunc[end_idx]:.2f}] N | "
        f"Peak R² = {r2_scores[linear_blocks].max():.4f}"
    )

    # Diagnostic plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_trunc, l_trunc, lw=1.5, label="Load–Time")
    ax.scatter(t_trunc[linear_blocks], l_trunc[linear_blocks],
               c='crimson', s=12, zorder=5, label="Linear region")
    ax.axvline(t_s, color='green',  ls='--', lw=1.5, label="Isolated window")
    ax.axvline(t_e, color='green',  ls='--', lw=1.5)
    ax.set(xlabel="Time (s)", ylabel="Load (N)",
           title="Hookean Regime Isolation — R² Analysis")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_path = save_path or os.path.splitext(UTM_FILE)[0] + "_Hookean.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    return t_s, t_e


def _auto_scale(series: pd.Series, target_max: float) -> float:
    """
    Return scale factor such that series.max() * scale ≈ target_max.
    Falls back to 1.0 if series is constant.
    """
    max_val = float(series.max())
    return (target_max / max_val) if abs(max_val) > 1e-8 else 1.0


# =============================================================================
#  DATA INGESTION — process_synchronized_tensors
# =============================================================================

def process_synchronized_tensors(
    dic_path: str,
    utm_path: str | None,
    checkpoint_norm: Normalizer | None = None,
    subsample_ratio: float = SUBSAMPLE_RATIO,
) -> tuple:
    """
    Load, synchronize, and pre-process the DIC + UTM data for DHC training.

    Returns
    -------
    x_n, y_n, t_n : normalized coordinate tensors   [N, 1]
    u_n, v_n      : normalized displacement tensors  [N, 1]
    exx, eyy, exy : raw strain tensors               [N, 1]
    p, p_dot      : load + load-rate tensors         [N, 1]
    bottom_mask, top_mask : boundary boolean masks   [N]
    norm          : fitted Normalizer
    anchor_xy_phys : physical (x, y) positions of DIC anchors for plotting [(K,2)]
    t_phys_range  : (t_min, t_max) in physical seconds
    """

    # ------------------------------------------------------------------ UTM --
    if utm_path is not None and os.path.exists(utm_path):
        utm_df   = pd.read_csv(utm_path, skiprows=[1])
        utm_df   = utm_df.apply(pd.to_numeric, errors='coerce').dropna()
        utm_time = utm_df['Time'].values
        utm_load = savgol_filter(utm_df['Load'].values, window_length=51, polyorder=3)
        load_dot = np.gradient(utm_load, np.mean(np.diff(utm_time)))
        t_start, t_end = isolate_hookean_regime(utm_time, utm_load)
    else:
        warnings.warn("UTM file not found — skipping Hookean isolation & force BCs.", RuntimeWarning)
        utm_time = utm_load = load_dot = None
        t_start, t_end = None, None

    # ------------------------------------------------------------------ DIC --
    df = pd.read_csv(dic_path)
    required = {'img_no', 'x_phys', 'y_phys', 'u_train', 'v_train',
                'exx_AVG', 'eyy_AVG', 'exy_AVG'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DIC CSV missing columns: {missing}")

    # Stride subsampling
    dic_df = df[df['img_no'] % STRIDE_FACTOR == 0].reset_index(drop=True)
    dic_df['Absolute_Time'] = dic_df['img_no'] / DIC_FRAME_RATE

    # Temporal isolation (Hookean window)
    n_frames_before = dic_df['img_no'].nunique()
    if t_start is not None:
        dic_df = dic_df[
            (dic_df['Absolute_Time'] >= t_start) &
            (dic_df['Absolute_Time'] <= t_end)
        ].reset_index(drop=True)
    n_frames_after = dic_df['img_no'].nunique()
    print(f"[DIC] Frame reduction: {n_frames_before} → {n_frames_after} (Hookean window)")

    # Temporal smoothing of displacements
    dic_df = smooth_displacement_field(dic_df)

    # Spatial subsampling (row-level)
    if subsample_ratio < 1.0:
        n_before = len(dic_df)
        dic_df   = dic_df.sample(frac=subsample_ratio, random_state=42).reset_index(drop=True)
        print(f"[DIC] Spatial subsample: {n_before} → {len(dic_df)} rows")

    # ----------------------------------------------------------------
    #  COORDINATE CALIBRATION
    #  The processed CSV uses arbitrary units; map to physical mm.
    # ----------------------------------------------------------------
    sx = COORD_SCALE_X if COORD_SCALE_X is not None else _auto_scale(dic_df['x_phys'], X_PHYS_MAX)
    sy = COORD_SCALE_Y if COORD_SCALE_Y is not None else _auto_scale(dic_df['y_phys'], Y_PHYS_MAX)

    x_phys = dic_df['x_phys'].values * sx   # mm
    y_phys = dic_df['y_phys'].values * sy   # mm

    # ----------------------------------------------------------------
    #  DISPLACEMENT CALIBRATION
    #  Enforce rigid-body subtraction: pin the bottom-most block.
    # ----------------------------------------------------------------
    ds = DISP_SCALE   # may be None initially

    # Bottom anchor: lowest ~15 % of y-range
    y_cutoff = float(np.percentile(y_phys, 15))
    bottom_rows = y_phys <= y_cutoff

    u_raw = dic_df['u_train'].values
    v_raw = dic_df['v_train'].values

    u_base = float(u_raw[bottom_rows].mean()) if bottom_rows.any() else 0.0
    v_base = float(v_raw[bottom_rows].mean()) if bottom_rows.any() else 0.0

    u_zeroed = u_raw - u_base
    v_zeroed = v_raw - v_base

    if ds is None:
        # Auto-scale so max |displacement| ≈ 10 mm (reasonable for a tensile test)
        max_disp = max(np.abs(u_zeroed).max(), np.abs(v_zeroed).max())
        ds = (10.0 / max_disp) if max_disp > 1e-8 else 1.0
        print(f"[Calibration] DISP_SCALE auto = {ds:.4e} | "
              f"COORD_SCALE_X = {sx:.4f} | COORD_SCALE_Y = {sy:.6f}")

    u_mm = u_zeroed * ds   # mm
    v_mm = v_zeroed * ds   # mm

    # ----------------------------------------------------------------
    #  STRAIN (dimensionless — keep as-is)
    # ----------------------------------------------------------------
    exx_raw = torch.tensor(dic_df['exx_AVG'].values, dtype=torch.float32).view(-1, 1)
    eyy_raw = torch.tensor(dic_df['eyy_AVG'].values, dtype=torch.float32).view(-1, 1)
    exy_raw = torch.tensor(dic_df['exy_AVG'].values, dtype=torch.float32).view(-1, 1)

    # ----------------------------------------------------------------
    #  BUILD RAW TENSORS
    # ----------------------------------------------------------------
    dic_time = dic_df['Absolute_Time'].values
    t_phys_range = (float(dic_time.min()), float(dic_time.max()))

    x_raw = torch.tensor(x_phys,            dtype=torch.float32).view(-1, 1)
    y_raw = torch.tensor(y_phys,            dtype=torch.float32).view(-1, 1)
    t_raw = torch.tensor(dic_time,          dtype=torch.float32).view(-1, 1)
    u_raw_t = torch.tensor(u_mm,            dtype=torch.float32).view(-1, 1)
    v_raw_t = torch.tensor(v_mm,            dtype=torch.float32).view(-1, 1)

    # ----------------------------------------------------------------
    #  LOAD SYNCHRONIZATION
    # ----------------------------------------------------------------
    if utm_time is not None:
        sync_load     = np.interp(dic_time, utm_time, utm_load)
        sync_load_dot = np.interp(dic_time, utm_time, load_dot)
    else:
        sync_load     = np.zeros_like(dic_time)
        sync_load_dot = np.zeros_like(dic_time)

    p_tensor     = torch.tensor(sync_load,     dtype=torch.float32).view(-1, 1)
    p_dot_tensor = torch.tensor(sync_load_dot, dtype=torch.float32).view(-1, 1)

    # ----------------------------------------------------------------
    #  NORMALIZATION
    # ----------------------------------------------------------------
    norm = checkpoint_norm if checkpoint_norm is not None else Normalizer()

    if checkpoint_norm is not None:
        x_n = norm.transform(x_raw, 'x')
        y_n = norm.transform(y_raw, 'y')
        t_n = norm.transform(t_raw, 't')
        u_n = norm.transform(u_raw_t, 'u')
        v_n = norm.transform(v_raw_t, 'v')
    else:
        x_n = norm.fit_transform(x_raw, 'x')
        y_n = norm.fit_transform(y_raw, 'y')
        t_n = norm.fit_transform(t_raw, 't')
        u_n = norm.fit_transform(u_raw_t, 'u')
        v_n = norm.fit_transform(v_raw_t, 'v')

    # ----------------------------------------------------------------
    #  BOUNDARY MASKS
    #  Bottom (Dirichlet — zero displacement), Top (Neumann — applied load)
    # ----------------------------------------------------------------
    physical_y = y_raw.squeeze()
    y_span     = float(physical_y.max() - physical_y.min())
    eps_frac   = 0.15  # ±15 % of y-span → boundary zone

    bottom_mask = physical_y <= float(physical_y.min()) + eps_frac * y_span
    top_mask    = physical_y >= float(physical_y.max()) - eps_frac * y_span

    if not bottom_mask.any() or not top_mask.any():
        raise ValueError(
            "Boundary masks are empty — check coordinate calibration. "
            f"y_phys range: [{float(physical_y.min()):.4f}, {float(physical_y.max()):.4f}] mm"
        )

    print(f"[Masks] Bottom: {bottom_mask.sum()} pts | Top: {top_mask.sum()} pts")

    # Unique spatial anchor positions (for plotting)
    anchor_xy_phys = np.unique(
        np.stack([x_phys, y_phys], axis=1), axis=0
    )

    return (
        x_n, y_n, t_n, u_n, v_n,
        exx_raw, eyy_raw, exy_raw,
        p_tensor, p_dot_tensor,
        bottom_mask, top_mask,
        norm, anchor_xy_phys, t_phys_range,
    )


# =============================================================================
#  DHC — GLOBAL COLLOCATION GENERATOR
# =============================================================================

def get_collocation_points(
    n: int,
    norm: Normalizer,
    device: torch.device,
    x_min: float = 0.0,
    x_max: float = X_PHYS_MAX,
    y_min: float = 0.0,
    y_max: float = Y_PHYS_MAX,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DHC core: generate a dense Sobol quasi-random cloud of (x, y, t) points
    spanning the full physical specimen domain [x_min, x_max] × [y_min, y_max].

    These points carry NO displacement labels. They are used exclusively to
    evaluate the Cauchy Momentum PDE residual, forcing the network to
    interpolate physically between the 8 sparse DIC anchors.

    Sobol sequences provide better space-filling coverage than pseudo-random
    sampling, which is critical for PDE residual convergence.
    """
    # Fall back to data time-range if not supplied
    t_min = t_min if t_min is not None else norm.mean['t'] - 3 * norm.std['t']
    t_max = t_max if t_max is not None else norm.mean['t'] + 3 * norm.std['t']

    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
    pts   = sobol.draw(n).to(device)     # ∈ [0, 1]³

    x_c = pts[:, 0:1] * (x_max - x_min) + x_min
    y_c = pts[:, 1:2] * (y_max - y_min) + y_min
    t_c = pts[:, 2:3] * (t_max - t_min) + t_min

    # Normalize to match the data coordinate space
    x_c_n = (x_c - norm.mean['x']) / norm.std['x']
    y_c_n = (y_c - norm.mean['y']) / norm.std['y']
    t_c_n = (t_c - norm.mean['t']) / norm.std['t']

    return (
        x_c_n.requires_grad_(True),
        y_c_n.requires_grad_(True),
        t_c_n.requires_grad_(True),
    )


# =============================================================================
#  NEURAL NETWORK ARCHITECTURE
# =============================================================================

class FourierFeatureEncoding(nn.Module):
    """Random Fourier Feature embedding for positional encoding."""

    def __init__(self, input_dim: int, mapping_size: int, scale: float = 1.0):
        super().__init__()
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    """Build a SiLU-activated MLP."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DHC_PINN(nn.Module):
    """
    Bifurcated PINN with separate kinematic and material branches.

    Kinematic branch  → u(x, y, t), v(x, y, t)
    Material branch   → E(x, y)  — time-invariant for a solid specimen
                        G derived analytically: G = E / (2(1+ν))

    The Fourier feature scales are deliberately different:
      • Kinematic encoder: higher scale → captures fine spatial & temporal variation
      • Material encoder : lower scale  → produces smooth E field (regularization)
    """

    def __init__(
        self,
        fourier_kin: int = FOURIER_KIN,
        fourier_mat: int = FOURIER_MAT,
    ):
        super().__init__()
        # Fourier encoders
        self.enc_kin = FourierFeatureEncoding(3, fourier_kin,       scale=0.01)
        self.enc_mat = FourierFeatureEncoding(3, fourier_mat // 2,  scale=0.005)

        feat_kin = fourier_kin * 2
        feat_mat = fourier_mat          # (fourier_mat//2) * 2

        # Kinematic branch
        self.kin_net = nn.Sequential(
            *_mlp(feat_kin, [256, 128], 2),  # outputs [u, v]
        )
        # Workaround: build as Sequential with proper final layer
        self.kin_trunk = _mlp(feat_kin, [256, 128], 128)
        self.kin_head  = nn.Linear(128, 2)

        # Material branch (spatial only — time frozen to zero)
        self.mat_trunk = _mlp(feat_mat, [128, 64, 32], 32)
        self.mat_head  = nn.Linear(32, 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = torch.cat([x, y, t], dim=1)

        # Kinematics
        kin_enc = self.enc_kin(coords)
        kin_lat = self.kin_trunk(kin_enc)
        kin_out = self.kin_head(kin_lat)
        u = kin_out[:, 0:1]
        v = kin_out[:, 1:2]

        # Material field — evaluate at spatial position only (t=0 stub)
        xy0 = torch.cat([x, y, torch.zeros_like(t)], dim=1)
        mat_enc = self.enc_mat(xy0)
        mat_lat = self.mat_trunk(mat_enc)
        E_raw   = self.mat_head(mat_lat)

        # Softplus ensures E > 0 and bounded below by E_REFERENCE/2
        E = (nn.functional.softplus(E_raw) + 0.5) * E_REFERENCE
        G = E / (2.0 * (1.0 + POISSONS_RATIO))

        return u, v, E, G


def initialize_weights(m: nn.Module) -> None:
    """Xavier-uniform init for SiLU networks; small positive bias."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


# =============================================================================
#  ADAPTIVE LAGRANGIAN WRAPPER
# =============================================================================

class AdaptiveLagrangianPINN(nn.Module):
    """
    Augmented Lagrangian wrapper.
    Log-space multipliers are optimized via gradient ascent (maximize=True).
    Ceiling clamp prevents pathological runaway.

    Multipliers:
      λ_pde    — Cauchy momentum residual (collocation)
      λ_force  — Global force boundary (dσ/dt ≈ dP/dt / A)
      λ_diric  — Dirichlet (bottom grip: u=v=0)
      λ_neum   — Neumann   (top grip: σ_yy ≈ P/A)
      λ_strain — Strain consistency at DIC anchors
      λ_smooth — E-field smoothness (Laplacian regularization)
    """
    LAMBDA_CEILING = 1e5

    def __init__(self, base: DHC_PINN):
        super().__init__()
        self.surrogate = base
        # Initial log-values chosen to produce O(1) multipliers
        self.log_l_pde    = nn.Parameter(torch.tensor([2.3]))
        self.log_l_force  = nn.Parameter(torch.tensor([7.0]))
        self.log_l_diric  = nn.Parameter(torch.tensor([6.9]))
        self.log_l_neum   = nn.Parameter(torch.tensor([6.9]))
        self.log_l_strain = nn.Parameter(torch.tensor([4.6]))
        self.log_l_smooth = nn.Parameter(torch.tensor([-4.6]))

    def forward(self, x, y, t):
        return self.surrogate(x, y, t)

    def get_multipliers(self) -> tuple:
        c = self.LAMBDA_CEILING
        return (
            torch.clamp(torch.exp(self.log_l_pde),    max=c),
            torch.clamp(torch.exp(self.log_l_force),  max=c),
            torch.clamp(torch.exp(self.log_l_diric),  max=c),
            torch.clamp(torch.exp(self.log_l_neum),   max=c),
            torch.clamp(torch.exp(self.log_l_strain), max=c),
            torch.clamp(torch.exp(self.log_l_smooth), max=c),
        )


# =============================================================================
#  LOSS FUNCTIONS
# =============================================================================

def _grad(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Safe autograd derivative; returns zeros if graph is disconnected."""
    g = torch.autograd.grad(
        out, inp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        allow_unused=True,
    )[0]
    return g if g is not None else torch.zeros_like(inp)


def compute_data_residuals(
    model: DHC_PINN,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    u_emp: torch.Tensor, v_emp: torch.Tensor,
    exx_emp: torch.Tensor, eyy_emp: torch.Tensor, exy_emp: torch.Tensor,
    p_emp: torch.Tensor, p_dot_emp: torch.Tensor,
    bottom_mask: torch.Tensor, top_mask: torch.Tensor,
    norm: Normalizer,
) -> tuple[torch.Tensor, ...]:
    """
    Compute residuals at DIC ANCHOR points only.
    Does NOT evaluate the Cauchy PDE — that is handled by compute_pde_residual().

    Returns
    -------
    loss_data     : displacement fit at 8 anchor blocks
    loss_strain   : strain consistency at anchor blocks (with noise-floor slack)
    loss_force    : global force rate boundary condition
    loss_dirichlet: Dirichlet u=v=0 at bottom grip
    loss_neumann  : Neumann σ_yy = P/A at top grip
    loss_smooth   : E-field Laplacian smoothness penalty
    """
    u_p, v_p, E_p, G_p = model(x, y, t)

    # Dimensional restoration scalars
    sx, sy = norm.std['x'], norm.std['y']
    st     = norm.std['t']
    su, sv = norm.std['u'], norm.std['v']

    # 1 — Displacement data term
    loss_data = torch.mean((u_p - u_emp) ** 2 + (v_p - v_emp) ** 2)

    # 2 — Kinematic strains at anchor locations
    eps_xx  = _grad(u_p, x) * (su / sx)
    eps_yy  = _grad(v_p, y) * (sv / sy)
    gam_xy  = _grad(u_p, y) * (su / sy) + _grad(v_p, x) * (sv / sx)

    raw_strain = torch.mean(
        (eps_xx - exx_emp) ** 2 +
        (eps_yy - eyy_emp) ** 2 +
        (gam_xy - exy_emp) ** 2
    )
    loss_strain = nn.functional.relu(raw_strain - EPS_STRAIN)

    # 3 — Constitutive stresses (plane stress)
    factor    = E_p / (1.0 - POISSONS_RATIO ** 2)
    sigma_xx  = factor * (eps_xx + POISSONS_RATIO * eps_yy)
    sigma_yy  = factor * (eps_yy + POISSONS_RATIO * eps_xx)
    tau_xy    = G_p * gam_xy

    # 4 — Global force rate BC: d(σ_yy·A)/dt ≈ dP/dt
    dsyy_dt   = _grad(sigma_yy, t) / st
    loss_force = torch.mean((dsyy_dt * CROSS_SECTIONAL_AREA - p_dot_emp) ** 2)

    # 5 — E-field smoothness (Tikhonov regularization of material field)
    loss_smooth = torch.mean(
        (_grad(E_p, x) / sx) ** 2 +
        (_grad(E_p, y) / sy) ** 2
    )

    # 6 — Dirichlet (bottom grip, u = v = 0 in physical space)
    if bottom_mask.any():
        u_phys_bot = u_p[bottom_mask] * su + norm.mean['u']
        v_phys_bot = v_p[bottom_mask] * sv + norm.mean['v']
        loss_dirichlet = torch.mean(u_phys_bot ** 2 + v_phys_bot ** 2)
    else:
        loss_dirichlet = torch.zeros(1, device=x.device)

    # 7 — Neumann (top grip: σ_yy ≈ applied traction)
    if top_mask.any():
        traction = p_emp[top_mask] / CROSS_SECTIONAL_AREA
        loss_neumann = torch.mean((sigma_yy[top_mask] - traction) ** 2)
    else:
        loss_neumann = torch.zeros(1, device=x.device)

    return loss_data, loss_strain, loss_force, loss_dirichlet, loss_neumann, loss_smooth


def compute_pde_residual(
    model: DHC_PINN,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    t_c: torch.Tensor,
    norm: Normalizer,
) -> torch.Tensor:
    """
    DHC core: evaluate the 2-D Cauchy Momentum residual at COLLOCATION points.

    Equation (plane stress + optional quasi-static inertia):
        ∂σ_xx/∂x + ∂τ_xy/∂y − ρ ü = 0
        ∂τ_xy/∂x + ∂σ_yy/∂y − ρ v̈ = 0

    With MATERIAL_DENSITY = 0, this reduces to static equilibrium:
        ∇·σ = 0

    These points carry NO displacement data — the network must satisfy the PDE
    entirely via the constitutive model E(x, y) it has inferred from the anchors.
    This is the mechanism that prevents non-physical extrapolation.
    """
    u_c, v_c, E_c, G_c = model(x_c, y_c, t_c)

    sx, sy = norm.std['x'], norm.std['y']
    st     = norm.std['t']
    su, sv = norm.std['u'], norm.std['v']

    eps_xx = _grad(u_c, x_c) * (su / sx)
    eps_yy = _grad(v_c, y_c) * (sv / sy)
    gam_xy = _grad(u_c, y_c) * (su / sy) + _grad(v_c, x_c) * (sv / sx)

    factor   = E_c / (1.0 - POISSONS_RATIO ** 2)
    sigma_xx = factor * (eps_xx + POISSONS_RATIO * eps_yy)
    sigma_yy = factor * (eps_yy + POISSONS_RATIO * eps_xx)
    tau_xy   = G_c * gam_xy

    res_x = _grad(sigma_xx, x_c) / sx + _grad(tau_xy, y_c) / sy
    res_y = _grad(tau_xy,   x_c) / sx + _grad(sigma_yy, y_c) / sy

    if MATERIAL_DENSITY > 0.0:
        u_tt = _grad(_grad(u_c, t_c), t_c) * (su / st ** 2)
        v_tt = _grad(_grad(v_c, t_c), t_c) * (sv / st ** 2)
        res_x = res_x - MATERIAL_DENSITY * u_tt
        res_y = res_y - MATERIAL_DENSITY * v_tt

    return torch.mean(res_x ** 2 + res_y ** 2)


# =============================================================================
#  CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    model_wrapper: AdaptiveLagrangianPINN,
    optimizer_theta: torch.optim.Optimizer,
    optimizer_lambda: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    norm: Normalizer,
    history: dict,
    loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'epoch'                   : epoch,
        'model_state_dict'        : model_wrapper.state_dict(),
        'optimizer_theta_state'   : optimizer_theta.state_dict(),
        'optimizer_lambda_state'  : optimizer_lambda.state_dict(),
        'scheduler_state'         : scheduler.state_dict(),
        'normalizer_mean'         : norm.mean,
        'normalizer_std'          : norm.std,
        'log_l_pde'               : model_wrapper.log_l_pde.detach().cpu(),
        'log_l_force'             : model_wrapper.log_l_force.detach().cpu(),
        'log_l_diric'             : model_wrapper.log_l_diric.detach().cpu(),
        'log_l_neum'              : model_wrapper.log_l_neum.detach().cpu(),
        'log_l_strain'            : model_wrapper.log_l_strain.detach().cpu(),
        'log_l_smooth'            : model_wrapper.log_l_smooth.detach().cpu(),
        'loss'                    : loss,
        'history'                 : history,
    }, path)


def load_checkpoint(path: str, device: torch.device) -> dict | None:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


# =============================================================================
#  TRAINING LOOP
# =============================================================================

def train(
    model_wrapper: AdaptiveLagrangianPINN,
    optimizer_theta: torch.optim.Optimizer,
    optimizer_lambda: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader: DataLoader,
    norm: Normalizer,
    device: torch.device,
    start_epoch: int,
    t_phys_range: tuple[float, float],
    history: dict,
    t_wall_start: float,
) -> None:
    """
    DHC training loop — two concurrent loss streams per step:

      Stream A (DATA)      : mini-batch from DIC dataloader
          → displacement fit + strain + BCs + smoothness
      Stream B (COLLOCATION): fresh Sobol sample from full domain
          → Cauchy PDE residual only
    """

    t_min_phys, t_max_phys = t_phys_range

    # Pre-generate collocation cloud (regenerated every RESAMPLE_COLLOC_EVERY epochs)
    x_c, y_c, t_c = get_collocation_points(
        N_COLLOC, norm, device,
        t_min=t_min_phys, t_max=t_max_phys,
    )

    # Collocation mini-batcher
    colloc_idx = torch.randperm(N_COLLOC, device=device)
    n_colloc_batches = max(1, N_COLLOC // BATCH_SIZE_COLLOC)

    for epoch in range(start_epoch, EPOCHS):
        model_wrapper.train()

        # Resample collocation cloud periodically for better coverage
        if epoch > start_epoch and epoch % RESAMPLE_COLLOC_EVERY == 0:
            x_c, y_c, t_c = get_collocation_points(
                N_COLLOC, norm, device,
                t_min=t_min_phys, t_max=t_max_phys,
            )
            colloc_idx = torch.randperm(N_COLLOC, device=device)

        epoch_stats = {k: 0.0 for k in
                       ['lagrangian', 'data', 'strain', 'pde', 'force', 'diric', 'neum', 'smooth']}
        n_batches = 0

        for batch_data in dataloader:
            (x_b, y_b, t_b, u_b, v_b,
             exx_b, eyy_b, exy_b,
             p_b, p_dot_b,
             bot_b, top_b) = [d.to(device) for d in batch_data]

            # TensorDataset stores bool masks as float; restore for indexing
            bot_mask_b = bot_b.squeeze().bool()
            top_mask_b = top_b.squeeze().bool()

            x_b.requires_grad_(True)
            y_b.requires_grad_(True)
            t_b.requires_grad_(True)

            optimizer_theta.zero_grad()
            optimizer_lambda.zero_grad()

            # --- Stream A: Data residuals ---
            l_data, l_strain, l_force, l_diric, l_neum, l_smooth = compute_data_residuals(
                model_wrapper.surrogate,
                x_b, y_b, t_b,
                u_b, v_b, exx_b, eyy_b, exy_b,
                p_b, p_dot_b, bot_mask_b, top_mask_b,
                norm,
            )

            # --- Stream B: PDE residual at a Sobol mini-batch ---
            batch_no    = n_batches % n_colloc_batches
            ci          = colloc_idx[batch_no * BATCH_SIZE_COLLOC:
                                      (batch_no + 1) * BATCH_SIZE_COLLOC]
            l_pde = compute_pde_residual(
                model_wrapper.surrogate,
                x_c[ci], y_c[ci], t_c[ci],
                norm,
            )

            # --- Adaptive Lagrangian objective ---
            lp, lf, ld, ln, ls, lsm = model_wrapper.get_multipliers()

            lagrangian = (
                LAMBDA_DATA * l_data
                + ls  * l_strain
                + lp  * l_pde
                + lf  * l_force
                + 10.0 * ld * l_diric
                + ln  * l_neum
                + lsm * l_smooth
            )

            lagrangian.backward()
            torch.nn.utils.clip_grad_norm_(
                model_wrapper.surrogate.parameters(), max_norm=1.0
            )
            optimizer_theta.step()

            # Multiplier ascent (every 5 data batches to prevent oscillation)
            if n_batches % 5 == 0:
                optimizer_lambda.step()

            epoch_stats['lagrangian'] += lagrangian.item()
            epoch_stats['data']       += l_data.item()
            epoch_stats['strain']     += l_strain.item()
            epoch_stats['pde']        += l_pde.item()
            epoch_stats['force']      += l_force.item()
            epoch_stats['diric']      += l_diric.item()
            epoch_stats['neum']       += l_neum.item()
            epoch_stats['smooth']     += l_smooth.item()
            n_batches += 1

        n = max(1, n_batches)
        avgs = {k: v / n for k, v in epoch_stats.items()}

        scheduler.step(avgs['lagrangian'])

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                lp, lf, ld, ln, ls, lsm = model_wrapper.get_multipliers()

            for k, v in avgs.items():
                history.setdefault(k, []).append(v)
            history.setdefault('epoch', []).append(epoch)
            history.setdefault('l_pde',    []).append(lp.item())
            history.setdefault('l_force',  []).append(lf.item())
            history.setdefault('l_diric',  []).append(ld.item())
            history.setdefault('l_neum',   []).append(ln.item())
            history.setdefault('l_strain', []).append(ls.item())
            history.setdefault('l_smooth', []).append(lsm.item())

            elapsed = time.time() - t_wall_start
            print(
                f"Epoch [{epoch:>4}/{EPOCHS}] "
                f"| L={avgs['lagrangian']:.3e} "
                f"| data={avgs['data']:.3e} "
                f"| pde={avgs['pde']:.3e} "
                f"| strain={avgs['strain']:.3e} "
                f"| smooth={avgs['smooth']:.3e} "
                f"|| λ_pde={lp.item():.2e} λ_d={ld.item():.2e} λ_s={ls.item():.2e} "
                f"| t={elapsed:.0f}s"
            )

        save_checkpoint(
            CHECKPOINT_FILE, epoch,
            model_wrapper, optimizer_theta, optimizer_lambda, scheduler,
            norm, history, avgs['lagrangian'],
        )

    print("Training complete.")


# =============================================================================
#  PHYSICAL PLAUSIBILITY CHECK
# =============================================================================

def physical_plausibility_check(
    model: DHC_PINN,
    norm: Normalizer,
    device: torch.device,
    n_probe: int = 500,
) -> None:
    """
    Sample random interior points and print E-field statistics.
    Warns if predicted E is outside the physically expected window.
    """
    model.eval()
    with torch.no_grad():
        x_p = torch.rand(n_probe, 1, device=device) * X_PHYS_MAX
        y_p = torch.rand(n_probe, 1, device=device) * Y_PHYS_MAX
        t_p = torch.full((n_probe, 1), norm.mean['t'], device=device)

        x_n = (x_p - norm.mean['x']) / norm.std['x']
        y_n = (y_p - norm.mean['y']) / norm.std['y']
        t_n = (t_p - norm.mean['t']) / norm.std['t']

        _, _, E_pred, _ = model(x_n, y_n, t_n)
        E_np = E_pred.cpu().numpy().flatten()

    e_mean, e_std = E_np.mean(), E_np.std()
    e_min, e_max = E_np.min(), E_np.max()
    print(
        f"[Plausibility] E field: mean={e_mean:.1f} MPa | "
        f"std={e_std:.1f} MPa | range=[{e_min:.1f}, {e_max:.1f}] MPa"
    )
    if e_min < 50 or e_max > 50_000:
        warnings.warn(
            f"E_pred range [{e_min:.1f}, {e_max:.1f}] MPa is outside the expected window "
            f"[50, 50 000] MPa. Check DISP_SCALE / COORD_SCALE calibration.",
            RuntimeWarning,
        )


# =============================================================================
#  VISUALIZATION
# =============================================================================

def _load_inference_model(
    weights_file: str,
    device: torch.device,
) -> tuple[DHC_PINN, dict, dict]:
    chk = load_checkpoint(weights_file, device)
    raw_sd = chk['model_state_dict']
    clean_sd = {k.replace('surrogate.', ''): v for k, v in raw_sd.items()}
    model = DHC_PINN().to(device)
    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    return model, chk['normalizer_mean'], chk['normalizer_std']


def _build_inference_grid(
    model: DHC_PINN,
    norm_mean: dict, norm_std: dict,
    x_limits: tuple, y_limits: tuple,
    query_time: float,
    grid_resolution: int,
    device: torch.device,
    requires_grad: bool = False,
):
    x = np.linspace(x_limits[0], x_limits[1], grid_resolution)
    y = np.linspace(y_limits[0], y_limits[1], grid_resolution)
    mesh_x, mesh_y = np.meshgrid(x, y)

    tx = torch.tensor(mesh_x.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    ty = torch.tensor(mesh_y.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    tt = torch.full_like(tx, query_time)

    if requires_grad:
        tx.requires_grad_(True); ty.requires_grad_(True)

    tx_n = (tx - norm_mean['x']) / norm_std['x']
    ty_n = (ty - norm_mean['y']) / norm_std['y']
    tt_n = (tt - norm_mean['t']) / norm_std['t']

    return mesh_x, mesh_y, tx, ty, tx_n, ty_n, tt_n


def generate_spatial_maps(
    weights_file: str,
    anchor_xy: np.ndarray,
    x_limits: tuple = (0, X_PHYS_MAX),
    y_limits: tuple = (0, Y_PHYS_MAX),
    query_time: float = 100.0,
    grid_resolution: int = 400,
    directory: str = OUTPUT_DIR,
) -> None:
    """
    Render constitutive and displacement field maps.
    Overlays DIC anchor positions as scatter markers for validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(directory, exist_ok=True)
    model, nm, ns = _load_inference_model(weights_file, device)

    mesh_x, mesh_y, _, _, tx_n, ty_n, tt_n = _build_inference_grid(
        model, nm, ns, x_limits, y_limits, query_time, grid_resolution, device
    )

    with torch.no_grad():
        u_t, v_t, E_t, G_t = model(tx_n, ty_n, tt_n)

    R = grid_resolution
    U  = (u_t * ns['u'] + nm['u']).cpu().numpy().reshape(R, R)
    V  = (v_t * ns['v'] + nm['v']).cpu().numpy().reshape(R, R)
    EM =  E_t.cpu().numpy().reshape(R, R)
    GM =  G_t.cpu().numpy().reshape(R, R)

    def _cplot(ax, mesh_x, mesh_y, Z, title, cmap, anchor_xy):
        c = ax.contourf(mesh_x, mesh_y, Z, levels=80, cmap=cmap)
        plt.colorbar(c, ax=ax, fraction=0.046)
        # Overlay DIC anchor positions
        ax.scatter(anchor_xy[:, 0], anchor_xy[:, 1],
                   c='white', edgecolors='black', s=60, zorder=5,
                   linewidths=0.8, label='DIC anchors')
        ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)')
        ax.legend(fontsize=8, loc='upper right')

    # --- Plot 1: Displacements ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    _cplot(axs[0], mesh_x, mesh_y, U, f'u-Displacement (mm) | t={query_time}', 'coolwarm', anchor_xy)
    _cplot(axs[1], mesh_x, mesh_y, V, f'v-Displacement (mm) | t={query_time}', 'coolwarm', anchor_xy)
    plt.tight_layout()
    plt.savefig(f'{directory}/DHC_Displacement_t{query_time}.png', dpi=300)
    plt.close()

    # --- Plot 2: Moduli ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    _cplot(axs[0], mesh_x, mesh_y, EM, 'Elastic Modulus E(x,y) [MPa]', 'viridis', anchor_xy)
    _cplot(axs[1], mesh_x, mesh_y, GM, 'Shear Modulus G(x,y) [MPa]',   'plasma',  anchor_xy)
    plt.tight_layout()
    plt.savefig(f'{directory}/DHC_Moduli_t{query_time}.png', dpi=300)
    plt.close()

    e_mean = float(EM.mean())
    print(f"[Output] Maps saved → {directory} | Mean E = {e_mean:.1f} MPa")


def generate_physics_maps(
    weights_file: str,
    anchor_xy: np.ndarray,
    x_limits: tuple = (0, X_PHYS_MAX),
    y_limits: tuple = (0, Y_PHYS_MAX),
    query_time: float = 100.0,
    grid_resolution: int = 300,
    directory: str = OUTPUT_DIR,
) -> None:
    """
    Compute and plot strain, stress, Von Mises, and PDE residual fields.
    PDE residual map acts as a self-diagnostic: it should be near-zero everywhere.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(directory, exist_ok=True)
    model, nm, ns = _load_inference_model(weights_file, device)

    mesh_x, mesh_y, tx, ty, tx_n, ty_n, tt_n = _build_inference_grid(
        model, nm, ns, x_limits, y_limits, query_time, grid_resolution, device,
        requires_grad=True,
    )

    u_t, v_t, E_t, G_t = model(tx_n, ty_n, tt_n)
    u_phys = u_t * ns['u'] + nm['u']
    v_phys = v_t * ns['v'] + nm['v']

    def ag(out, inp):
        g = torch.autograd.grad(out, inp, torch.ones_like(out), retain_graph=True)[0]
        return g if g is not None else torch.zeros_like(inp)

    Exx = ag(u_phys, tx).cpu().detach().numpy().reshape(grid_resolution, grid_resolution)
    Eyy = ag(v_phys, ty).cpu().detach().numpy().reshape(grid_resolution, grid_resolution)
    Gxy = (ag(u_phys, ty) + ag(v_phys, tx)).cpu().detach().numpy().reshape(grid_resolution, grid_resolution)

    EM = E_t.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
    GM = G_t.detach().cpu().numpy().reshape(grid_resolution, grid_resolution)
    nu = POISSONS_RATIO
    Sxx = (EM / (1 - nu**2)) * (Exx + nu * Eyy)
    Syy = (EM / (1 - nu**2)) * (Eyy + nu * Exx)
    Txy = GM * Gxy
    VM  = np.sqrt(Sxx**2 - Sxx * Syy + Syy**2 + 3 * Txy**2)

    def _cplot2(ax, mesh_x, mesh_y, Z, title, cmap):
        c = ax.contourf(mesh_x, mesh_y, Z, levels=60, cmap=cmap)
        plt.colorbar(c, ax=ax, fraction=0.046)
        ax.scatter(anchor_xy[:, 0], anchor_xy[:, 1],
                   c='white', edgecolors='k', s=50, zorder=5)
        ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)')

    # Strain fields
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    _cplot2(axs[0], mesh_x, mesh_y, Exx, 'ε_xx (Longitudinal strain)', 'inferno')
    _cplot2(axs[1], mesh_x, mesh_y, Eyy, 'ε_yy (Transverse strain)',   'inferno')
    _cplot2(axs[2], mesh_x, mesh_y, Gxy, 'γ_xy (Shear strain)',        'inferno')
    plt.tight_layout()
    plt.savefig(f'{directory}/DHC_Strains_t{query_time}.png', dpi=300)
    plt.close()

    # Stress + Von Mises
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    _cplot2(axs[0], mesh_x, mesh_y, Syy, 'σ_yy [MPa] (Axial stress)',  'magma')
    _cplot2(axs[1], mesh_x, mesh_y, Sxx, 'σ_xx [MPa] (Lateral stress)', 'magma')
    _cplot2(axs[2], mesh_x, mesh_y, VM,  'Von Mises Stress [MPa]',      'magma')
    plt.tight_layout()
    plt.savefig(f'{directory}/DHC_Stress_t{query_time}.png', dpi=300)
    plt.close()

    print(f"[Output] Physics maps saved → {directory}")


def plot_training_history(history: dict, directory: str = OUTPUT_DIR) -> None:
    """Plot loss component trajectories and Lagrangian multiplier evolution."""
    os.makedirs(directory, exist_ok=True)
    epochs = history.get('epoch', [])
    if not epochs:
        return

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    panels = [
        ('lagrangian', 'Augmented Lagrangian', 'tab:blue'),
        ('data',       'Data Loss (anchors)',   'tab:green'),
        ('pde',        'PDE Residual (DHC)',    'tab:red'),
        ('strain',     'Strain Consistency',    'tab:orange'),
        ('smooth',     'E Smoothness',          'tab:purple'),
    ]
    for k, (key, title, color) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        if key in history:
            ax.semilogy(epochs, history[key], color=color, lw=1.5)
        ax.set(title=title, xlabel='Epoch', ylabel='Loss')
        ax.grid(True, alpha=0.3)

    # Multiplier panel
    ax_m = fig.add_subplot(gs[1, 2])
    for key, label in [('l_pde', 'λ_PDE'), ('l_diric', 'λ_Diric'),
                        ('l_strain', 'λ_Strain'), ('l_smooth', 'λ_Smooth')]:
        if key in history:
            ax_m.semilogy(epochs, history[key], label=label, lw=1.2)
    ax_m.set(title='Lagrange Multipliers', xlabel='Epoch', ylabel='Value')
    ax_m.legend(fontsize=8)
    ax_m.grid(True, alpha=0.3)

    plt.suptitle("DHC-PINN Training History", fontsize=13, fontweight='bold')
    plt.savefig(f'{directory}/DHC_Training_History.png', dpi=200)
    plt.close()
    print(f"[Output] Training history plot saved.")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    t_wall = time.time()
    print(f"[Start] {time.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Guard: check data files ---
    if not os.path.exists(DIC_FILE):
        raise FileNotFoundError(f"DIC file not found: {DIC_FILE}")

    utm_available = os.path.exists(UTM_FILE)
    if not utm_available:
        warnings.warn(f"UTM file not found ({UTM_FILE}) — force BCs disabled.", RuntimeWarning)

    # --- Load data (with optional checkpoint normalizer for resuming) ---
    existing_chk = load_checkpoint(CHECKPOINT_FILE, device)

    if existing_chk is not None:
        saved_norm = Normalizer()
        saved_norm.mean = existing_chk['normalizer_mean']
        saved_norm.std  = existing_chk['normalizer_std']
        print(f"[Checkpoint] Found at epoch {existing_chk['epoch']} — resuming.")
    else:
        saved_norm = None
        print("[Checkpoint] None found — initialising from scratch.")

    (X, Y, T, U, V,
     Exx, Eyy, Exy,
     P, P_dot,
     bot_mask, top_mask,
     norm, anchor_xy, t_range) = process_synchronized_tensors(
        DIC_FILE,
        UTM_FILE if utm_available else None,
        checkpoint_norm=saved_norm,
        subsample_ratio=SUBSAMPLE_RATIO,
    )

    print(f"[Dataset] {len(X)} training rows | "
          f"{anchor_xy.shape[0]} unique DIC anchor positions | "
          f"t ∈ [{t_range[0]:.3f}, {t_range[1]:.3f}] s")

    # --- DataLoader (anchor data only — Sobol colloc generated inside loop) ---
    dataset    = TensorDataset(X, Y, T, U, V, Exx, Eyy, Exy, P, P_dot,
                               bot_mask.float().view(-1, 1),
                               top_mask.float().view(-1, 1))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_DATA,
                            shuffle=True, drop_last=False, pin_memory=False)

    # Note: bot_mask / top_mask are stored as float32 in TensorDataset
    # (TensorDataset cannot store bool in a batch-compatible way).
    # They are re-cast to bool inside the training loop before indexing.

    # --- Model & Optimizers ---
    base_model    = DHC_PINN(fourier_kin=FOURIER_KIN, fourier_mat=FOURIER_MAT)
    base_model.apply(initialize_weights)
    model_wrapper = AdaptiveLagrangianPINN(base_model).to(device)

    optimizer_theta = torch.optim.Adam(
        model_wrapper.surrogate.parameters(), lr=LR_THETA
    )
    optimizer_lambda = torch.optim.Adam(
        [model_wrapper.log_l_pde,   model_wrapper.log_l_force,
         model_wrapper.log_l_diric, model_wrapper.log_l_neum,
         model_wrapper.log_l_strain, model_wrapper.log_l_smooth],
        lr=LR_LAMBDA, maximize=True,
    )
    # ReduceLROnPlateau: halve LR if Lagrangian stalls for 150 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_theta, mode='min', factor=0.5, patience=150)#, verbose=True,)

    # --- Restore from checkpoint ---
    start_epoch = 0
    history     = {}
    if existing_chk is not None:
        chk = existing_chk
        try:
            model_wrapper.load_state_dict(chk['model_state_dict'])
        except RuntimeError:
            model_wrapper.surrogate.load_state_dict(
                {k.replace('surrogate.', ''): v for k, v in chk['model_state_dict'].items()},
                strict=False,
            )
        if 'optimizer_theta_state' in chk:
            optimizer_theta.load_state_dict(chk['optimizer_theta_state'])
            optimizer_lambda.load_state_dict(chk['optimizer_lambda_state'])
        if 'scheduler_state' in chk:
            scheduler.load_state_dict(chk['scheduler_state'])
        start_epoch = chk['epoch'] + 1
        history     = chk.get('history', {})

    # --- Training ---
    train(
        model_wrapper, optimizer_theta, optimizer_lambda, scheduler,
        dataloader, norm, device,
        start_epoch, t_range, history, t_wall,
    )

    # --- Post-training diagnostics ---
    physical_plausibility_check(model_wrapper.surrogate, norm, device)
    plot_training_history(history, directory=OUTPUT_DIR)

    # Infer query_time as midpoint of training window
    query_t = float(np.mean(t_range))
    query_t_ = query_t + float(np.std(t_range) * 0.5)
    query__t = query_t - float(np.std(t_range) * 0.5)

    generate_spatial_maps(
        CHECKPOINT_FILE, anchor_xy,
        x_limits=(0, X_PHYS_MAX),
        y_limits=(0, Y_PHYS_MAX),
        query_time=query_t,
        grid_resolution=400,
        directory=OUTPUT_DIR,
    )

    generate_spatial_maps(
        CHECKPOINT_FILE, anchor_xy,
        x_limits=(0, X_PHYS_MAX),
        y_limits=(0, Y_PHYS_MAX),
        query_time=query_t_,
        grid_resolution=400,
        directory=OUTPUT_DIR,
    )
    generate_spatial_maps(
        CHECKPOINT_FILE, anchor_xy,
        x_limits=(0, X_PHYS_MAX),
        y_limits=(0, Y_PHYS_MAX),
        query_time=query__t,
        grid_resolution=300,
        directory=OUTPUT_DIR,
    )

    generate_physics_maps(
        CHECKPOINT_FILE, anchor_xy,
        x_limits=(0, X_PHYS_MAX),
        y_limits=(0, Y_PHYS_MAX),
        query_time=query_t,
        grid_resolution=300,
        directory=OUTPUT_DIR,
    )

    generate_physics_maps(
        CHECKPOINT_FILE, anchor_xy,
        x_limits=(0, X_PHYS_MAX),
        y_limits=(0, Y_PHYS_MAX),
        query_time=query_t_,
        grid_resolution=300,
        directory=OUTPUT_DIR,
    )

    generate_physics_maps(
        CHECKPOINT_FILE, anchor_xy,
        x_limits=(0, X_PHYS_MAX),
        y_limits=(0, Y_PHYS_MAX),
        query_time=query__t,
        grid_resolution=300,
        directory=OUTPUT_DIR,
    )

    print(f"\n[Done] Total elapsed: {time.time() - t_wall:.1f}s")
