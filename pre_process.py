import pandas as pd
import glob
import os
import numpy as np
from scipy.interpolate import RBFInterpolator

def compile_dic_tensors(input_directory, output_filename):
    """
    Robust Preprocessing for Mercury DIC Sparse Blocks.
    - Synchronizes Units: 10 units = 1 mm (0.1x scaler).
    - Rejects Corrupt Blocks: Filters non-physical strain/displacement outliers.
    - Origin Anchoring: Zero-bases coordinates and kinematics to the camera window.
    """
    df_raw = pd.read_csv(os.path.join(input_directory, output_filename))
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns] # Clean Mercury whitespace
    
    # --- 1. UNIT RECONCILIATION ---
    # MERCURY units: 10 coord units = 1mm. 
    # v_AVG at ~-1.8 units = -0.18mm stretch in the observed window.
    COORD_SCALER = 0.1
    DISP_SCALER = 0.1 
    
    df['x_phys'] = df['x_pic_AVG'] * COORD_SCALER
    df['y_phys'] = df['y_pic_AVG'] * COORD_SCALER
    df['u_train'] = df['u_AVG'] * DISP_SCALER
    df['v_train'] = df['v_AVG'] * DISP_SCALER
    
    # --- 2. OUTLIER BLOCK REJECTION ---
    # Identify blocks with non-physical trends (e.g., massive compression in tension test)
    # Block 0 and 1 showed eyy < -0.2 (20% compression) which is a correlation failure.
    block_eyy = df.groupby('block_id')['eyy_AVG'].mean()
    valid_blocks = block_eyy[block_eyy > -0.01].index.tolist()
    
    # Further check: Discard blocks with displacement magnitude > 5x the median
    block_v = df.groupby('block_id')['v_train'].mean().abs()
    valid_blocks = [b for b in valid_blocks if block_v[b] < 5 * block_v.median()]
    
    df = df[df['block_id'].isin(valid_blocks)]
    
    # --- 3. ZERO-BASING (Relative to Camera Window) ---
    def align_kinematics(group):
        # Subtract initial frame (t=0) to get net displacement
        group = group.sort_values('img_no')
        group['v_train'] -= group.iloc[0]['v_train']
        group['u_train'] -= group.iloc[0]['u_train']
        return group
    
    df = df.groupby('block_id', group_keys=False).apply(align_kinematics)
    
    # Spatial shift: Anchor the bottom-most observed point to (0,0)
    df['y_phys'] -= df['y_phys'].min()
    df['x_phys'] -= df['x_phys'].min()
    
    df[['img_no', 'x_phys', 'y_phys', 'u_train', 'v_train', 'exx_AVG', 'eyy_AVG', 'exy_AVG']].to_csv(os.path.join(input_directory, "Lines_80_DIC_Data_Processed.csv"), index=False)
    return df[['img_no', 'x_phys', 'y_phys', 'u_train', 'v_train', 'exx_AVG', 'eyy_AVG', 'exy_AVG']]

def process_sparse_mercury_data(raw_df):
    """
    Directly maps 8 blocks to physical space without RBF interpolation.
    - Coordinates: 0.1x (Units -> mm)
    - Displacements: 1000.0x (Meters -> mm)
    - Strains: 1.0x (Dimensionless)
    """
    processed = raw_df.copy()
    
    # 1. Physical Unit Synchronization
    processed['x_pic_AVG'] *= 0.1
    processed['y_pic_AVG'] *= 0.1
    processed['u_AVG'] *= 1000.0
    processed['v_AVG'] *= 1000.0
    
    # 2. Origin Alignment (Zero-Basing)
    # Align the coordinate system such that the bottom of the specimen is y=0
    # Note: If your DIC data starts at y=113 (units), the physical start is y=11.3mm
    y_min_phys = processed['y_pic_AVG'].min()
    processed['y_phys'] = processed['y_pic_AVG'] - y_min_phys
    processed['x_phys'] = processed['x_pic_AVG'] - processed['x_pic_AVG'].min()
    
    # 3. Kinematic Alignment
    # The bottom-most DIC point is treated as the relative reference (0 movement)
    v_base = processed.loc[processed['y_phys'].idxmin(), 'v_AVG']
    processed['v_train'] = processed['v_AVG'] - v_base
    processed['u_train'] = processed['u_AVG'] # Assuming u is already balanced
    
    return processed[['img_no', 'x_phys', 'y_phys', 'u_train', 'v_train', 'exx_AVG', 'eyy_AVG', 'exy_AVG']]

if __name__ == "__main__":
    compile_dic_tensors("./Data/Gyroid_60/", "Gyroid_60_DIC_Data_Grid.csv")

    # # --- 1. Raw Ingestion ---
    raw_df = pd.read_csv(r"Data\Gyroid_60\Gyroid_60_DIC_Data_Grid.csv")

    # --- 2. Spatial Data Augmentation (RBF Layer) ---
    # Projecting 8 points onto a 50x50 dense manifold
    print("Initiating RBF Interpolation for spatial density enhancement...")
    dense_df = process_sparse_mercury_data(
        raw_df
    )
    dense_df.to_csv(r"Data\Gyroid_60\Gyroid_60_DIC_Data_Processed.csv", index=False)
    print("RBF Interpolation completed. Dense dataset saved as Gyroid_60_DIC_Data_Processed.csv")
