"""
propagate_dic_to_2d.py
======================
Takes your real single-spatial-point DIC CSV and UTM CSV and produces a
physics-consistent 2D grid DIC file ready for the PINN.

Usage
-----
    python propagate_dic_to_2d.py
        --dic  _DIC_Data_Grid_1.csv
        --utm  Specimen_RawData_1.csv
        --out  DIC_2D_Data.csv
        --nx   7
        --ny   10

Memory-safe: rows are written to disk in chunks — never accumulates the
full dataset in RAM.
"""

import argparse
import csv
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Material defaults (Aluminium) ─────────────────────────────────────────────
E_DEFAULT    = 70000.0
NU           = 0.3
G_DEFAULT    = E_DEFAULT / (2 * (1 + NU))
SCF_AMP      = 0.12       # +/-12 % stress-concentration amplitude across width
NOISE_DISP   = 2e-4       # mm — DIC displacement noise std
NOISE_STRAIN = 3e-5       # strain noise std

CHUNK_SIZE   = 200        # frames per write-to-disk batch


def load_utm(path: str) -> pd.DataFrame:
    utm = pd.read_csv(path, skiprows=[1])
    utm = utm.apply(pd.to_numeric, errors='coerce').dropna()
    utm.columns = [c.strip() for c in utm.columns]
    wl = min(51, len(utm) // 2 * 2 - 1)
    filt = savgol_filter(utm['Load'].values, window_length=wl, polyorder=3)
    dt = np.mean(np.diff(utm['Time'].values))
    utm['Load_filt'] = filt
    utm['Load_dot']  = np.gradient(filt, dt)
    utm['Load_ddot'] = np.gradient(utm['Load_dot'].values, dt)
    return utm


def load_dic(path: str) -> pd.DataFrame:
    dic = pd.read_csv(path)
    dic = dic.apply(pd.to_numeric, errors='coerce')
    dic.columns = [c.strip() for c in dic.columns]
    return dic


def stress_concentration(x: np.ndarray, x_mid: float, width: float) -> np.ndarray:
    return 1.0 + SCF_AMP * np.exp(-((x - x_mid) ** 2) / (width / 4) ** 2)


def propagate_frame(row: pd.Series, nx: int, ny: int,
                    rng: np.random.Generator) -> list:
    x_avg = float(row['x_pic_AVG'])
    x_min = float(row['x_pic_Min'])
    x_max = float(row['x_pic_Max'])
    y_avg = float(row['y_pic_AVG'])
    y_min = float(row['y_pic_Min'])
    y_max = float(row['y_pic_Max'])

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys)
    XX = XX.ravel()
    YY = YY.ravel()
    n = len(XX)

    x_mid = (x_min + x_max) / 2.0
    y_bot = y_min
    width = max(x_max - x_min, 1e-6)

    exx_m = float(row['exx_AVG'])
    eyy_m = float(row['eyy_AVG'])
    exy_m = float(row['exy_AVG'])
    u_m   = float(row['u_AVG'])
    v_m   = float(row['v_AVG'])

    scf        = stress_concentration(XX, x_mid, width)
    exx_field  = exx_m * scf;  exx_field  -= exx_field.mean()  - exx_m
    eyy_field  = eyy_m * scf;  eyy_field  -= eyy_field.mean()  - eyy_m
    exy_field  = exy_m + rng.normal(0, NOISE_STRAIN, n)

    u_field = exx_field * (XX - x_mid) + (exy_field / 2) * (YY - y_bot)
    v_field = eyy_field * (YY - y_bot) + (exy_field / 2) * (XX - x_mid)
    u_field += u_m - u_field.mean()
    v_field += v_m - v_field.mean()
    u_field += rng.normal(0, NOISE_DISP, n)
    v_field += rng.normal(0, NOISE_DISP, n)
    exx_field += rng.normal(0, NOISE_STRAIN, n)
    eyy_field += rng.normal(0, NOISE_STRAIN, n)

    disp_mag = np.sqrt(u_field**2 + v_field**2)
    avg_n    = (exx_field + eyy_field) / 2
    R        = np.sqrt(((exx_field - eyy_field) / 2)**2 + (exy_field / 2)**2)
    e1       = avg_n + R
    e2       = avg_n - R
    gamma    = 2 * exy_field
    evm      = np.sqrt(np.clip(e1**2 - e1*e2 + e2**2, 0, None))

    f00 = 1.0 + exx_field + rng.normal(0, 5e-5, n)
    f01 = exy_field / 2    + rng.normal(0, 5e-5, n)
    f10 = exy_field / 2    + rng.normal(0, 5e-5, n)
    f11 = 1.0 + eyy_field  + rng.normal(0, 5e-5, n)

    r_avg   = float(row.get('r_AVG',   0.99997))
    sig_avg = float(row.get('sigma_AVG', 7e-5))
    cp_avg  = float(row.get('correlationPath_AVG', 30))

    r_f  = np.clip(rng.normal(r_avg,  2e-5, n), 0.999, 1.0)
    sf   = np.abs(rng.normal(sig_avg, 1e-5, n))
    cp_f = np.clip(rng.normal(cp_avg, 3.0,  n), 1, 48).round()

    ns = NOISE_DISP
    ne = NOISE_STRAIN
    dx = (x_max - x_min) / max(nx - 1, 1) / 2
    dy = (y_max - y_min) / max(ny - 1, 1) / 2

    rows = []
    for i in range(n):
        rows.append({
            'img_no':               f"{float(row['img_no']):.5E}",
            'x_pic_AVG':            f"{XX[i]:.5E}",
            'x_pic_Min':            f"{XX[i]-dx:.5E}",
            'x_pic_Max':            f"{XX[i]+dx:.5E}",
            'y_pic_AVG':            f"{YY[i]:.5E}",
            'y_pic_Min':            f"{YY[i]-dy:.5E}",
            'y_pic_Max':            f"{YY[i]+dy:.5E}",
            'u_AVG':                f"{u_field[i]:.5E}",
            'u_Min':                f"{u_field[i]-2*ns:.5E}",
            'u_Max':                f"{u_field[i]+2*ns:.5E}",
            'v_AVG':                f"{v_field[i]:.5E}",
            'v_Min':                f"{v_field[i]-2*ns:.5E}",
            'v_Max':                f"{v_field[i]+2*ns:.5E}",
            'disp_magnitude_AVG':   f"{disp_mag[i]:.5E}",
            'disp_magnitude_Min':   f"{max(0,disp_mag[i]-2*ns):.5E}",
            'disp_magnitude_Max':   f"{disp_mag[i]+2*ns:.5E}",
            'r_AVG':                f"{r_f[i]:.5E}",
            'r_Min':                f"{r_f[i]-3e-5:.5E}",
            'r_Max':                f"{r_f[i]+1e-5:.5E}",
            'sigma_AVG':            f"{sf[i]:.5E}",
            'sigma_Min':            f"{sf[i]*0.7:.5E}",
            'sigma_Max':            f"{sf[i]*1.3:.5E}",
            'correlationPath_AVG':  f"{cp_f[i]:.5E}",
            'correlationPath_Min':  f"{1.0:.5E}",
            'correlationPath_Max':  f"{48.0:.5E}",
            'ShapeIndex_AVG':       f"{float(row.get('ShapeIndex_AVG',0.0)):.5E}",
            'ShapeIndex_Min':       f"{0.0:.5E}",
            'ShapeIndex_Max':       f"{0.0:.5E}",
            'exx_AVG':              f"{exx_field[i]:.5E}",
            'exx_Min':              f"{exx_field[i]-2*ne:.5E}",
            'exx_Max':              f"{exx_field[i]+2*ne:.5E}",
            'eyy_AVG':              f"{eyy_field[i]:.5E}",
            'eyy_Min':              f"{eyy_field[i]-2*ne:.5E}",
            'eyy_Max':              f"{eyy_field[i]+2*ne:.5E}",
            'exy_AVG':              f"{exy_field[i]:.5E}",
            'exy_Min':              f"{exy_field[i]-ne:.5E}",
            'exy_Max':              f"{exy_field[i]+ne:.5E}",
            'e1_AVG':               f"{e1[i]:.5E}",
            'e1_Min':               f"{e1[i]-2*ne:.5E}",
            'e1_Max':               f"{e1[i]+2*ne:.5E}",
            'e2_AVG':               f"{e2[i]:.5E}",
            'e2_Min':               f"{e2[i]-2*ne:.5E}",
            'e2_Max':               f"{e2[i]+2*ne:.5E}",
            'gamma_AVG':            f"{gamma[i]:.5E}",
            'gamma_Min':            f"{gamma[i]-2*ne:.5E}",
            'gamma_Max':            f"{gamma[i]+2*ne:.5E}",
            'evm_AVG':              f"{evm[i]:.5E}",
            'evm_Min':              f"{max(0,evm[i]-2*ne):.5E}",
            'evm_Max':              f"{evm[i]+2*ne:.5E}",
            'f00_AVG':              f"{f00[i]:.5E}",
            'f00_Min':              f"{f00[i]-5e-4:.5E}",
            'f00_Max':              f"{f00[i]+5e-4:.5E}",
            'f01_AVG':              f"{f01[i]:.5E}",
            'f01_Min':              f"{f01[i]-3e-4:.5E}",
            'f01_Max':              f"{f01[i]+3e-4:.5E}",
            'f10_AVG':              f"{f10[i]:.5E}",
            'f10_Min':              f"{f10[i]-3e-4:.5E}",
            'f10_Max':              f"{f10[i]+3e-4:.5E}",
            'f11_AVG':              f"{f11[i]:.5E}",
            'f11_Min':              f"{f11[i]-5e-4:.5E}",
            'f11_Max':              f"{f11[i]+5e-4:.5E}",
        })
    return rows


COLUMNS = [
    'Index', 'img_no',
    'x_pic_AVG', 'x_pic_Min', 'x_pic_Max',
    'y_pic_AVG', 'y_pic_Min', 'y_pic_Max',
    'u_AVG', 'u_Min', 'u_Max',
    'v_AVG', 'v_Min', 'v_Max',
    'disp_magnitude_AVG', 'disp_magnitude_Min', 'disp_magnitude_Max',
    'r_AVG', 'r_Min', 'r_Max',
    'sigma_AVG', 'sigma_Min', 'sigma_Max',
    'correlationPath_AVG', 'correlationPath_Min', 'correlationPath_Max',
    'ShapeIndex_AVG', 'ShapeIndex_Min', 'ShapeIndex_Max',
    'exx_AVG', 'exx_Min', 'exx_Max',
    'eyy_AVG', 'eyy_Min', 'eyy_Max',
    'exy_AVG', 'exy_Min', 'exy_Max',
    'e1_AVG', 'e1_Min', 'e1_Max',
    'e2_AVG', 'e2_Min', 'e2_Max',
    'gamma_AVG', 'gamma_Min', 'gamma_Max',
    'evm_AVG', 'evm_Min', 'evm_Max',
    'f00_AVG', 'f00_Min', 'f00_Max',
    'f01_AVG', 'f01_Min', 'f01_Max',
    'f10_AVG', 'f10_Min', 'f10_Max',
    'f11_AVG', 'f11_Min', 'f11_Max',
]


def main():
    parser = argparse.ArgumentParser(description='Propagate single-point DIC to 2D grid.')
    parser.add_argument('--dic',  default='_DIC_Data_Grid_1.csv',  help='DIC input CSV')
    parser.add_argument('--utm',  default='Specimen_RawData_1.csv', help='UTM input CSV')
    parser.add_argument('--out',  default='DIC_2D_Data.csv',        help='Output CSV path')
    parser.add_argument('--nx',   type=int, default=5,              help='Grid points in x')
    parser.add_argument('--ny',   type=int, default=25,             help='Grid points in y')
    parser.add_argument('--seed', type=int, default=42,             help='Random seed')
    args = parser.parse_args()

    print(f"Loading DIC : {args.dic}")
    dic_df = load_dic(args.dic)
    n_frames = len(dic_df)
    pts_per_frame = args.nx * args.ny
    print(f"  {n_frames} frames -> {n_frames * pts_per_frame:,} output rows "
          f"({args.nx} x {args.ny} grid)")

    print(f"Loading UTM : {args.utm}")
    load_utm(args.utm)

    rng = np.random.default_rng(args.seed)
    global_index = 0

    with open(args.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()

        chunk_buf = []
        flush_every = CHUNK_SIZE * pts_per_frame

        for frame_idx, row in dic_df.iterrows():
            pts = propagate_frame(row, args.nx, args.ny, rng)

            for pt in pts:
                pt['Index'] = global_index
                global_index += 1
                chunk_buf.append(pt)

            if len(chunk_buf) >= flush_every:
                writer.writerows(chunk_buf)
                chunk_buf.clear()

            if (frame_idx + 1) % 100 == 0 or frame_idx == n_frames - 1:
                print(f"  [{frame_idx+1:>5}/{n_frames}] frames done "
                      f"-- {global_index:,} rows written")

        if chunk_buf:
            writer.writerows(chunk_buf)

    print(f"\nDone. {global_index:,} rows saved to '{args.out}'")


if __name__ == '__main__':
    main()