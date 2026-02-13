"""
NFL Preprocessing Pipeline - v10 (Feature Test)

This script handles all data preprocessing with advanced v2 features.
Run this first before training.

Usage:
    python nfl_preprocessing_v10.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = ""
PROCESSED_DATA_DIR = Path("working/processed_data")
TRAIN_INPUT_GLOB = "../train/input_2023_w*.csv"
TRAIN_OUTPUT_GLOB = "../train/output_2023_w*.csv"

TRAIN_WEEKS = list(range(1, 17))  # 1-16
VAL_WEEKS = list(range(17, 19))   # 17-18

LOOKBACK = 25
FIELD_X_MAX = 120.0
FIELD_Y_MAX = 53.3

# ============================================================================
# ADVANCED FEATURE ENGINEERING (from v2)
# ============================================================================

def compute_physics_features(df):
    """Tier 1: Basic Physics"""
    df = df.copy()
    df['fe__vx'] = df['s'] * np.cos(np.deg2rad(df['dir']))
    df['fe__vy'] = df['s'] * np.sin(np.deg2rad(df['dir']))
    
    # Acceleration components
    df['fe__ax'] = df['a'] * np.cos(np.deg2rad(df['dir']))
    df['fe__ay'] = df['a'] * np.sin(np.deg2rad(df['dir']))
    
    # Trajectory curvature
    dir_change = df.groupby(['game_id', 'play_id', 'nfl_id'])['dir'].diff()
    df['fe__traj_curvature'] = dir_change.abs() / (df['s'] + 1e-3)
    df['fe__traj_curvature'] = df['fe__traj_curvature'].fillna(0)
    
    # Deceleration proxy
    angle_diff = np.deg2rad(df['o'] - df['dir'])
    df['fe__decel_proxy'] = df['a'] * np.cos(angle_diff)
    return df

def compute_interaction_features(df):
    """Tier 2: Multi-Player Interactions"""
    df = df.copy()
    df['fe__pursuit_angle'] = 0.0
    df['fe__vel_convergence'] = 0.0
    df['fe__def_coverage_density'] = 0.0
    
    # Use computed ball_current positions
    if 'ball_current_x' not in df.columns:
        df['ball_current_x'] = df.get('ball_land_x', 50.0)
    if 'ball_current_y' not in df.columns:
        df['ball_current_y'] = df.get('ball_land_y', 26.65)

    grouped = df.groupby(['game_id', 'play_id', 'frame_id'])

    for (game_id, play_id, frame_id), frame_df in grouped:
        if len(frame_df) < 2:
            continue

        positions = frame_df[['x', 'y']].values
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        closest_idx = np.argmin(dist_matrix, axis=1)

        for i, (idx, row) in enumerate(frame_df.iterrows()):
            closest_row = frame_df.iloc[closest_idx[i]]
            dy = closest_row['y'] - row['y']
            dx = closest_row['x'] - row['x']
            angle_to_closest = np.degrees(np.arctan2(dy, dx))
            pursuit_angle = (angle_to_closest - row['dir'] + 180) % 360 - 180
            df.loc[idx, 'fe__pursuit_angle'] = pursuit_angle

            dist_to_closest = dist_matrix[i, closest_idx[i]]
            if dist_to_closest > 0:
                closing_speed = row['s'] * np.cos(np.deg2rad(pursuit_angle))
                vel_conv = closing_speed / dist_to_closest
                df.loc[idx, 'fe__vel_convergence'] = np.clip(vel_conv, -10, 10)

        ball_pos = frame_df[['ball_current_x', 'ball_current_y']].iloc[0].values
        player_pos = frame_df[['x', 'y']].values
        dist_to_ball = np.linalg.norm(player_pos - ball_pos, axis=1)
        inv_dist = 1.0 / (dist_to_ball + 1e-6)
        total_density = np.sum(inv_dist)
        df.loc[frame_df.index, 'fe__def_coverage_density'] = total_density
    return df

def compute_context_features(df):
    """Tier 3: Play Context & Embeddings"""
    df = df.copy()
    period = 10
    df['fe__route_freq_sin'] = np.sin(2 * np.pi * df['frame_id'] / period) * df['s']
    df['fe__route_freq_cos'] = np.cos(2 * np.pi * df['frame_id'] / period) * df['s']
    df['fe__formation_cluster'] = 0

    route_features = df[['fe__traj_curvature', 's']].fillna(0)
    if len(route_features) > 10:
        try:
            n_clusters = min(8, len(route_features) - 1)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
                df['fe__route_cluster'] = kmeans.fit_predict(route_features)
            else:
                df['fe__route_cluster'] = 0
        except Exception:
            df['fe__route_cluster'] = 0
    else:
        df['fe__route_cluster'] = 0
    return df

def create_zone_coverage_features(df):
    """Model defensive zone responsibilities"""
    df = df.copy()
    
    zones = {
        'deep_left': {'x': (60, 120), 'y': (0, 17.8)},
        'deep_middle': {'x': (60, 120), 'y': (17.8, 35.5)},
        'deep_right': {'x': (60, 120), 'y': (35.5, 53.3)},
        'short_left': {'x': (40, 60), 'y': (0, 17.8)},
        'short_middle': {'x': (40, 60), 'y': (17.8, 35.5)},
        'short_right': {'x': (40, 60), 'y': (35.5, 53.3)},
    }
    
    for zone_name, bounds in zones.items():
        x_in_zone = (df['x'] >= bounds['x'][0]) & (df['x'] <= bounds['x'][1])
        y_in_zone = (df['y'] >= bounds['y'][0]) & (df['y'] <= bounds['y'][1])
        df[f'fe__zone_{zone_name}'] = (x_in_zone & y_in_zone).astype(float)
        
        zone_center_x = np.mean(bounds['x'])
        zone_center_y = np.mean(bounds['y'])
        df[f'fe__dist_to_{zone_name}'] = np.sqrt(
            (df['x'] - zone_center_x)**2 + (df['y'] - zone_center_y)**2
        )
    
    return df
def add_temporal_derivative_features(df):
    """
    Add temporal derivative features (jerk, angular velocity, etc.)
    """
    df = df.copy()
    
    # Group by player within each play
    grouped = df.groupby(['game_id', 'play_id', 'nfl_id'])
    
    # 1. Jerk (change in acceleration)
    df['fe__jerk_magnitude'] = grouped['a'].diff().fillna(0)
    
    # 2. Angular velocity (change in direction)
    df['fe__angular_velocity'] = grouped['dir'].diff().fillna(0)
    # Handle wrap-around (e.g., 359° -> 1°)
    df.loc[df['fe__angular_velocity'] > 180, 'fe__angular_velocity'] -= 360
    df.loc[df['fe__angular_velocity'] < -180, 'fe__angular_velocity'] += 360
    
    # 3. Speed change rate
    df['fe__speed_change_rate'] = grouped['s'].diff().fillna(0) * 10  # per second
    
    # 4. Orientation change rate  
    df['fe__orientation_change_rate'] = grouped['o'].diff().fillna(0)
    df.loc[df['fe__orientation_change_rate'] > 180, 'fe__orientation_change_rate'] -= 360
    df.loc[df['fe__orientation_change_rate'] < -180, 'fe__orientation_change_rate'] += 360
    
    # 5. Trajectory consistency (difference between dir and o)
    df['fe__dir_o_diff'] = (df['dir'] - df['o'] + 180) % 360 - 180
    
    return df


def engineer_features(df):
    """Enhanced feature engineering with ALL v2 features"""
    df = df.copy()
    
    required_cols_defaults = {
        'x': 0.0, 'y': 0.0, 's': 0.0, 'a': 0.0, 'o': 0.0, 'dir': 0.0,
        'frame_id': 0, 'game_id': 0, 'play_id': 0, 'nfl_id': 0,
        'ball_land_x': 50.0, 'ball_land_y': 26.65,
        'num_frames_output': 25.0
    }
    for col, default_val in required_cols_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Basic features
    df['fe__vx'] = df['s'] * np.cos(np.deg2rad(df['dir']))
    df['fe__vy'] = df['s'] * np.sin(np.deg2rad(df['dir']))
    dx = df['ball_land_x'] - df['x']
    dy = df['ball_land_y'] - df['y']
    df['fe__distance_to_target'] = np.sqrt(dx**2 + dy**2)
    
    # BALL TRAJECTORY MODELING
    df['ball_distance'] = df['fe__distance_to_target']
    df['estimated_ball_flight_time'] = df['ball_distance'] / 24.0
    df['time_since_throw'] = df['frame_id'] * 0.1
    df['ball_progress'] = np.clip(df['time_since_throw'] / (df['estimated_ball_flight_time'] + 0.1), 0, 1)
    
    df['ball_current_x'] = df['x'] + (df['ball_land_x'] - df['x']) * df['ball_progress']
    df['ball_current_y'] = df['y'] + (df['ball_land_y'] - df['y']) * df['ball_progress']
    df['ball_height'] = 4 * df['ball_progress'] * (1 - df['ball_progress']) * 8
    
    df['fe__dist_to_ball_3d'] = np.sqrt(
        (df['ball_current_x'] - df['x'])**2 + 
        (df['ball_current_y'] - df['y'])**2 + 
        df['ball_height']**2
    )
    
    time_to_ball_land = np.maximum(0, df['estimated_ball_flight_time'] - df['time_since_throw'])
    max_distance_can_cover = df['s'] * time_to_ball_land * 1.2
    df['fe__can_reach_ball'] = (df['fe__distance_to_target'] <= max_distance_can_cover).astype(float)
    
    # PURSUIT ANGLES
    if 's' in df.columns and df['s'].mean() > 0:
        time_to_intercept = df['fe__distance_to_target'] / (df['s'] + 0.1)
        future_ball_x = df['x'] + (df['ball_land_x'] - df['x']) * np.minimum(1, time_to_intercept / 2.5)
        future_ball_y = df['y'] + (df['ball_land_y'] - df['y']) * np.minimum(1, time_to_intercept / 2.5)
        
        opt_pursuit_angle = np.rad2deg(np.arctan2(
            future_ball_y - df['y'],
            future_ball_x - df['x']
        ))
        df['fe__optimal_pursuit_angle'] = (opt_pursuit_angle - df['dir'] + 180) % 360 - 180
    else:
        df['fe__optimal_pursuit_angle'] = 0
    
    # FIELD POSITION CONTEXT
    df['fe__in_red_zone'] = (df['x'] >= 100).astype(float)
    df['fe__dist_to_sideline'] = np.minimum(df['y'], 53.3 - df['y'])
    df['fe__field_third'] = pd.cut(df['x'], bins=[0, 40, 80, 120], labels=[0, 1, 2], include_lowest=True)
    
    # MOMENTUM FEATURES
    ball_dir_x = (df['ball_land_x'] - df['x']) / (df['fe__distance_to_target'] + 0.001)
    ball_dir_y = (df['ball_land_y'] - df['y']) / (df['fe__distance_to_target'] + 0.001)
    df['fe__momentum_toward_ball'] = df['fe__vx'] * ball_dir_x + df['fe__vy'] * ball_dir_y
    df['fe__closing_speed'] = df['fe__momentum_toward_ball'] / (df['fe__distance_to_target'] + 0.001)
    
    # COVERAGE METRICS
    df['fe__time_to_ball_spot'] = df['fe__distance_to_target'] / (df['s'] + 0.1)
    df['fe__good_position'] = (
        (np.abs(df['fe__optimal_pursuit_angle']) < 45) & 
        (df['fe__momentum_toward_ball'] > 0)
    ).astype(float)
    
    # Apply all feature tiers
    df = compute_physics_features(df)
    df = compute_interaction_features(df)
    df = compute_context_features(df)
    df = create_zone_coverage_features(df)
    df = add_temporal_derivative_features(df)
    
    return df

# ============================================================================
# BASE PREPROCESSING UTILITIES
# ============================================================================

def normalize_bool(val):
    if pd.isna(val): return False
    if isinstance(val, (bool, np.bool_)): return bool(val)
    s = str(val).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

def extract_week_from_filename(filepath):
    basename = os.path.basename(filepath)
    try:
        week_str = basename.split('_w')[1].split('.')[0]
        return int(week_str)
    except:
        return None

def safe_get_column(df, col, default=0.0):
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)

def identify_players_to_predict(df):
    df = df.copy()
    if 'player_role' in df.columns:
        df['player_to_predict'] = df['player_role'].str.contains('Defensive', na=False)
    elif 'position' in df.columns:
        defensive_positions = ['CB', 'FS', 'SS', 'LB', 'OLB', 'ILB', 'MLB', 'DB', 'S']
        df['player_to_predict'] = df['position'].isin(defensive_positions)
    else:
        df['player_to_predict'] = True
    return df

def create_categorical_features(df):
    df = df.copy()
    play_direction = safe_get_column(df, 'play_direction', 'right')
    df['play_direction_left'] = (play_direction.str.lower() == 'left').astype(float)
    return df

# ============================================================================
# PREPROCESSOR CLASS
# ============================================================================

class NFLPreprocessorWithTargets:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.all_features = numeric_features + categorical_features

        self.position_scaler = StandardScaler()
        self.other_scaler = StandardScaler()
        self.is_fitted = False

        self.position_features = ['x', 'y']
        self.other_features = [f for f in numeric_features if f not in self.position_features]

    def fit(self, df):
        for col in self.all_features:
            if col not in df.columns:
                df[col] = 0.0

        if self.position_features:
            position_data = df[self.position_features].fillna(0).values
            self.position_scaler.fit(position_data)

        if self.other_features:
            other_data = df[self.other_features].fillna(0).values
            self.other_scaler.fit(other_data)

        self.is_fitted = True
        return self

    def transform_inputs(self, df):
        for col in self.all_features:
            if col not in df.columns:
                df[col] = 0.0

        transformed_parts = []

        if self.position_features:
            position_data = df[self.position_features].fillna(0)
            position_transformed = self.position_scaler.transform(position_data)
            transformed_parts.append(position_transformed)

        if self.other_features:
            other_data = df[self.other_features].fillna(0)
            other_transformed = self.other_scaler.transform(other_data)
            other_transformed = np.clip(other_transformed, -10, 10)
            transformed_parts.append(other_transformed)

        if self.categorical_features:
            cat_data = df[self.categorical_features].fillna(0).values
            transformed_parts.append(cat_data)

        return np.hstack(transformed_parts)

    def transform_targets(self, targets):
        return self.position_scaler.transform(targets)

# ============================================================================
# PLAY PROCESSING
# ============================================================================

def process_play(play_input, play_output, passer_row, gt_ball_land_xy, preprocessor, feature_cols):
    """Packages a single play into a model-ready tensor dictionary."""
    if 'player_to_predict' not in play_input.columns:
        play_input = identify_players_to_predict(play_input)

    players = play_input[play_input['player_to_predict'] == True]['nfl_id'].unique()
    if len(players) == 0:
        players = play_input['nfl_id'].unique()

    # Throw features (8-dim)
    throw_features = np.array([
        passer_row.get('x', 50.0) / 120.0, 
        passer_row.get('y', 26.65) / 53.3,
        passer_row.get('ball_land_x', 50.0) / 120.0, 
        passer_row.get('ball_land_y', 26.65) / 53.3,
        np.cos(np.deg2rad(passer_row.get('dir', 0.0))), 
        np.sin(np.deg2rad(passer_row.get('dir', 0.0))),
        0.0, 0.0 
    ], dtype=np.float32)

    gt_ball_land_xy_scaled = preprocessor.transform_targets(gt_ball_land_xy.reshape(1, -1))[0]

    input_sequences, target_sequences, valid_players, player_positions = [], [], [], []

    for nfl_id in players:
        player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
        player_output = play_output[play_output['nfl_id'] == nfl_id].sort_values('frame_id')

        if len(player_output) == 0:
            continue

        target_raw = player_output[['x', 'y']].values
        if len(target_raw) == 0 or np.any(np.isnan(target_raw)):
            continue

        # Padding logic
        if len(player_input) >= LOOKBACK:
            player_input = player_input.tail(LOOKBACK)
        else:
            pad_len = LOOKBACK - len(player_input)
            pad_df = pd.DataFrame([player_input.iloc[0]] * pad_len)
            player_input = pd.concat([pad_df, player_input], ignore_index=True)

        features = preprocessor.transform_inputs(player_input[feature_cols].fillna(0))
        input_sequences.append(features)

        target_scaled = preprocessor.transform_targets(target_raw)
        target_sequences.append(target_scaled)
        valid_players.append(nfl_id)

        last_pos = player_input[['x', 'y']].iloc[-1].values
        player_positions.append(last_pos)

    if len(valid_players) == 0:
        return None

    num_players = len(valid_players)
    player_positions = np.array(player_positions)

    edges, weights = [], []

    for i in range(num_players):
        for j in range(num_players):
            if i != j:
                dist = np.linalg.norm(player_positions[i] - player_positions[j])
                if dist < 30.0:
                    weight = np.exp(-dist / 10.0)
                    edges.append([i, j])
                    weights.append(weight)

    if len(edges) == 0:
        edges, weights = [[0, 0]], [0.0]

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weights = torch.tensor(weights, dtype=torch.float32)

    max_len = max(len(t) for t in target_sequences)
    padded_targets = []

    for target in target_sequences:
        if len(target) < max_len:
            last_pos = target[-1] if len(target) > 0 else np.zeros(2)
            padding = np.tile(last_pos, (max_len - len(target), 1))
            target = np.vstack([target, padding])
        padded_targets.append(target)

    return {
        'input_sequence': torch.tensor(np.array(input_sequences), dtype=torch.float32),
        'target_sequence': torch.tensor(np.array(padded_targets), dtype=torch.float32),
        'throw_features': torch.tensor(throw_features, dtype=torch.float32),
        'gt_ball_land_xy': torch.tensor(gt_ball_land_xy_scaled, dtype=torch.float32),
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'player_mask': torch.ones(len(valid_players), dtype=torch.bool),
        'num_players': len(valid_players)
    }

# ============================================================================
# PARALLEL PROCESSING WORKER
# ============================================================================

def process_week_chunk(week, in_paths_map, out_paths_map, preprocessor, feature_cols, train_dir, val_dir):
    """Worker function for parallel processing of one week"""
    if week not in out_paths_map:
        return 0, 0, 0, 0

    try:
        dfi = pd.read_csv(in_paths_map[week])
        dfo = pd.read_csv(out_paths_map[week])

        # Run full feature engineering
        dfi = identify_players_to_predict(dfi)
        dfi = engineer_features(dfi)
        dfi = create_categorical_features(dfi)

        # Determine save directory
        if week in TRAIN_WEEKS:
            save_dir = train_dir
        elif week in VAL_WEEKS:
            save_dir = val_dir
        else:
            return 0, 0, 0, 0

        week_plays_data = []
        train_count_w, val_count_w, skipped_count_w, skipped_no_passer_w = 0, 0, 0, 0

        # Process each play
        for (game_id, play_id), play_group in dfi.groupby(['game_id', 'play_id']):
            passer_mask = play_group['player_role'] == 'Passer' if 'player_role' in play_group.columns else pd.Series([False] * len(play_group))
            passer_data = play_group[passer_mask]

            if len(passer_data) == 0:
                skipped_no_passer_w += 1
                skipped_count_w += 1
                continue

            passer_row = passer_data.iloc[0]
            gt_ball_land_xy = np.array([
                passer_row.get('ball_land_x', 50.0),
                passer_row.get('ball_land_y', 26.65)
            ], dtype=np.float32)

            input_group = play_group[play_group['player_to_predict'].apply(normalize_bool)].copy()
            output_group = dfo[
                (dfo['game_id'] == game_id) &
                (dfo['play_id'] == play_id)
            ]

            if len(output_group) == 0:
                skipped_count_w += 1
                continue

            try:
                play_data = process_play(input_group, output_group, passer_row,
                                        gt_ball_land_xy, preprocessor, feature_cols)

                if play_data is None:
                    skipped_count_w += 1
                    continue

                week_plays_data.append(play_data)

                if week in TRAIN_WEEKS:
                    train_count_w += 1
                elif week in VAL_WEEKS:
                    val_count_w += 1

            except Exception:
                skipped_count_w += 1
                continue

        # Save entire week's data as one file
        if week_plays_data:
            filename = f"week_{week:02d}.pt"
            torch.save(week_plays_data, save_dir / filename)
            print(f"   [Week {week:02d}] Saved {len(week_plays_data)} plays to {save_dir / filename}")

        return train_count_w, val_count_w, skipped_count_w, skipped_no_passer_w

    except Exception as e:
        print(f"Error processing week {week}: {e}")
        return 0, 0, 0, 0

# ============================================================================
# MAIN PREPROCESSING RUNNER
# ============================================================================

def run_preprocessing():
    """Run the complete preprocessing pipeline"""
    print("\n" + "="*80)
    print("🚀 NFL PREPROCESSING PIPELINE - v10")
    print("="*80)
    print("🆕 Using v2 Advanced Feature Engineering")
    print("✅ Zone Coverage Features")
    print("✅ Ball Trajectory Modeling")
    print("✅ Parallel Processing")
    print("="*80)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_dir = PROCESSED_DATA_DIR / "train"
    val_dir = PROCESSED_DATA_DIR / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Output directory: {PROCESSED_DATA_DIR}")

    # STEP 1: Build Preprocessor
    print("\n" + "="*80)
    print("STEP 1: Building Preprocessor")
    print("="*80)

    in_paths = sorted(glob.glob(TRAIN_INPUT_GLOB))
    sample_dfs = []
    for p in in_paths:
        dfi = pd.read_csv(p)
        dfi = identify_players_to_predict(dfi)
        dfi = dfi[dfi["player_to_predict"].apply(normalize_bool)].copy()
        sample_dfs.append(dfi)

    df_sample = pd.concat(sample_dfs, ignore_index=True)

    print("[INFO] Generating feature names...")
    df_sample_features = engineer_features(df_sample.head(100).copy())
    engineered_features = [c for c in df_sample_features.columns if c.startswith('fe__')]
    
    RAW_NUMERIC_FEATURES = [
        "x", "y", "s", "a", "dir", "o",
        "ball_land_x", "ball_land_y",
        "num_frames_output"
    ]
    
    CATEGORICAL_FEATURES = ["play_direction_left"]
    
    numeric_features = RAW_NUMERIC_FEATURES + engineered_features
    categorical_features = CATEGORICAL_FEATURES
    feature_cols = numeric_features + categorical_features

    print(f"[INFO] Fitting preprocessor... ({len(feature_cols)} features)")
    preprocessor = NFLPreprocessorWithTargets(numeric_features, categorical_features)
    preprocessor.fit(df_sample)
    print(f"[OK] Preprocessor fitted")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Engineered features: {len(engineered_features)}")

    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.pt"
    torch.save({
        'position_mean': preprocessor.position_scaler.mean_.tolist(),
        'position_scale': preprocessor.position_scaler.scale_.tolist(),
        'other_mean': preprocessor.other_scaler.mean_.tolist() if hasattr(preprocessor.other_scaler, 'mean_') else None,
        'other_scale': preprocessor.other_scaler.scale_.tolist() if hasattr(preprocessor.other_scaler, 'scale_') else None,
        'position_features': preprocessor.position_features,
        'other_features': preprocessor.other_features,
        'categorical_features': preprocessor.categorical_features,
        'feature_cols': feature_cols
    }, preprocessor_path)
    print(f"[OK] Preprocessor saved to {preprocessor_path}")

    # STEP 2: Parallel Processing
    print("\n" + "="*80)
    print("STEP 2: Parallel Processing")
    print(f"   Using {os.cpu_count()} cores...")
    print("="*80)

    in_paths_map = {extract_week_from_filename(p): p for p in sorted(glob.glob(TRAIN_INPUT_GLOB)) if p}
    out_paths_map = {extract_week_from_filename(p): p for p in sorted(glob.glob(TRAIN_OUTPUT_GLOB)) if p}
    all_weeks = sorted([k for k in in_paths_map.keys() if k is not None])

    results = Parallel(n_jobs=-1)(
        delayed(process_week_chunk)(
            week, in_paths_map, out_paths_map, preprocessor, feature_cols, train_dir, val_dir
        ) for week in tqdm(all_weeks, desc="Processing weeks")
    )

    # Aggregate results
    train_count = sum(r[0] for r in results if r)
    val_count = sum(r[1] for r in results if r)
    skipped_count = sum(r[2] for r in results if r)
    skipped_no_passer = sum(r[3] for r in results if r)

    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Train plays: {train_count}")
    print(f"   Val plays: {val_count}")
    print(f"   Skipped (no passer): {skipped_no_passer}")
    print(f"   Skipped (other): {skipped_count - skipped_no_passer}")
    print(f"   Total saved: {train_count + val_count}")

    print(f"\n💾 Saved to:")
    print(f"   {train_dir} ({len(list(train_dir.glob('week_*.pt')))} files)")
    print(f"   {val_dir} ({len(list(val_dir.glob('week_*.pt')))} files)")
    print(f"   {preprocessor_path}")
    print(f"\n🚀 Ready for training!")
    
    return preprocessor_path

if __name__ == "__main__":
    print("""
    ================================================================================
    🏈 NFL PREPROCESSING PIPELINE - v10
    ================================================================================
    Advanced feature engineering with v2 features on stable v9 pipeline
    - Zone coverage features
    - Ball trajectory modeling
    - Physics-based features
    - Multi-player interactions
    ================================================================================
    """)
    
    try:
        preprocessor_path = run_preprocessing()
        print(f"\n✅ SUCCESS! Preprocessor saved to: {preprocessor_path}")
        print("Next step: Run the training script")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
