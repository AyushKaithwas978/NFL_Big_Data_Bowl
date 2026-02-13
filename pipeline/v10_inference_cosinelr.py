"""
NFL GNN-Transformer: INFERENCE SCRIPT v10 CosineLoss
Matching v10_test1_preprocessing.py features and v10 training architecture

Training Configuration:
- LOOKBACK: 25 frames
- D_MODEL: 128
- N_HEADS: 8
- N_LAYERS: 3
- ENSEMBLE_SEEDS: [42]
- Features: Advanced v2 with temporal derivatives from v10_test1_preprocessing.py

Features:
✅ Zone Coverage Features
✅ Advanced Ball Trajectory Modeling  
✅ Physics-aware features
✅ Temporal derivative features (jerk, angular velocity, etc.)
✅ Adaptive IQR-based model filtering
✅ Weighted averaging with α=2.5 penalty
✅ TTA: Horizontal Flip + Rotation
✅ Post-processing: Smoothing + Speed constraints
"""

import os
import glob
import warnings
import json
import math
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")
import kaggle_evaluation.nfl_inference_server

# ============================================================================
# CONFIGURATION
# ============================================================================
MODELS_DIR = Path("/kaggle/input/cosineloss-test/pytorch/default/1")
PREPROCESSOR_PATH = Path("/kaggle/input/test1-preprocessor/pytorch/default/1/preprocessor.pt")

# Ensemble configuration - v10 ReduceOnPlateau uses seed 42 only
ENSEMBLE_SEEDS = [42]
LOOKBACK = 25  # v10 uses LOOKBACK=25
FIELD_X_MAX = 120.0
FIELD_Y_MAX = 53.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model architecture parameters (from v10 training)
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3

# Adaptive weighting parameters
ALPHA = 2.5  # Harsher penalty for worse models
MIN_WEIGHT = 0.05  # Minimum weight per model
MAX_WEIGHT = 0.30  # Maximum weight per model

# TTA Configuration
APPLY_TTA = True
APPLY_ROTATION_TTA = True
ROTATION_ANGLES = [5, -5]  # Rotation angles in degrees

# Post-processing Configuration
APPLY_SMOOTHING = True
APPLY_SPEED_CAP = True
SAVGOL_WINDOW = 5
SAVGOL_ORDER = 2
MAX_SPEED_YD_PER_SEC = 12.0
FRAME_RATE = 10.0

# Global variables
MODELS_AND_WEIGHTS = None
PREPROCESSOR_PARAMS = None

# ============================================================================
# MODEL DEFINITIONS (from v10 training)
# ============================================================================

class BallTrajectoryPredictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Dropout(0.1)
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), 
            nn.Linear(32, 3)
        )
    
    def forward(self, throw_features):
        feat = self.encoder(throw_features).unsqueeze(1)
        _, hidden = self.rnn(feat)
        return self.output(hidden[-1])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class OptimizedGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(in_features * 2, 1)
    
    def forward(self, node_features, edge_index, edge_weights, edge_mask):
        batch_size, num_nodes, in_feat = node_features.shape
        transformed = self.linear(node_features)
        aggregated = torch.zeros_like(transformed)
        
        for b in range(batch_size):
            valid_edges = edge_mask[b]
            if not valid_edges.any():
                continue
            
            edges = edge_index[b, :, valid_edges]
            weights = edge_weights[b, valid_edges]
            
            if edges.shape[1] == 0:
                continue
            
            src_nodes, tgt_nodes = edges[0], edges[1]
            src_feats = node_features[b, src_nodes]
            tgt_feats = node_features[b, tgt_nodes]
            
            concat = torch.cat([tgt_feats, src_feats], dim=-1)
            attn_scores = torch.sigmoid(self.attention(concat))
            messages = attn_scores * weights.unsqueeze(-1) * transformed[b, src_nodes]
            aggregated[b].index_add_(0, tgt_nodes, messages.to(aggregated[b].dtype))
        
        return transformed + aggregated

class HybridModel(nn.Module):
    """Main trajectory prediction model - matches v10 training exactly"""
    def __init__(self, input_dim, d_model=128, gnn_dim=128,
                 decoder_hidden=128, nhead=8, num_encoder_layers=3):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        self.ball_predictor = BallTrajectoryPredictor()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=LOOKBACK)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.gnn1 = OptimizedGraphConvLayer(d_model, gnn_dim)
        self.gnn2 = OptimizedGraphConvLayer(gnn_dim, gnn_dim)
        
        self.decoder_gru = nn.GRU(
            input_size=gnn_dim + 2 + 3,
            hidden_size=decoder_hidden,
            num_layers=2, batch_first=True, dropout=0.1
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(decoder_hidden, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 2)
        )
    
    def forward(self, input_sequence, throw_features, edge_index, 
                edge_weights, edge_mask, num_steps):
        batch_size, num_players, seq_len, features = input_sequence.shape
        assert features == self.input_dim, f"Input features {features} != model input_dim {self.input_dim}"
        assert seq_len == LOOKBACK, f"Sequence length {seq_len} != LOOKBACK {LOOKBACK}"
        
        pcp = self.ball_predictor(throw_features)
        
        x = input_sequence.reshape(batch_size * num_players, seq_len, features)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        context = x[:, -1, :].reshape(batch_size, num_players, self.d_model)
        context = F.relu(self.gnn1(context, edge_index, edge_weights, edge_mask))
        context = self.gnn2(context, edge_index, edge_weights, edge_mask)
        
        pcp_expanded = pcp[:, :2].unsqueeze(1).expand(-1, num_players, -1)
        context_flat = context.reshape(batch_size * num_players, -1)
        pcp_flat = pcp_expanded.reshape(batch_size * num_players, 2)
        pcp_conf = pcp[:, 2:3].unsqueeze(1).expand(-1, num_players, -1).reshape(-1, 1)
        
        predictions = []
        current_pos = pcp_flat.unsqueeze(1)
        hidden = None
        
        for _ in range(num_steps):
            decoder_input = torch.cat([
                current_pos, context_flat.unsqueeze(1),
                pcp_flat.unsqueeze(1), pcp_conf.unsqueeze(1)
            ], dim=-1)
            output, hidden = self.decoder_gru(decoder_input, hidden)
            next_pos = self.output_layer(output)
            predictions.append(next_pos)
            current_pos = next_pos
        
        predictions = torch.cat(predictions, dim=1)
        return predictions.reshape(batch_size, num_players, num_steps, 2), pcp

# ============================================================================
# FEATURE ENGINEERING (v10 Advanced Features - EXACTLY from v10_test1_preprocessing.py)
# ============================================================================

def compute_physics_features(df):
    """Tier 1: Basic Physics"""
    df = df.copy()
    df['fe__vx'] = df['s'] * np.cos(np.deg2rad(df['dir']))
    df['fe__vy'] = df['s'] * np.sin(np.deg2rad(df['dir']))
    
    df['fe__ax'] = df['a'] * np.cos(np.deg2rad(df['dir']))
    df['fe__ay'] = df['a'] * np.sin(np.deg2rad(df['dir']))
    
    dir_change = df.groupby(['game_id', 'play_id', 'nfl_id'])['dir'].diff()
    df['fe__traj_curvature'] = dir_change.abs() / (df['s'] + 1e-3)
    df['fe__traj_curvature'] = df['fe__traj_curvature'].fillna(0)
    
    angle_diff = np.deg2rad(df['o'] - df['dir'])
    df['fe__decel_proxy'] = df['a'] * np.cos(angle_diff)
    
    return df

def compute_interaction_features(df):
    """Tier 2: Multi-Player Interactions"""
    df = df.copy()
    df['fe__pursuit_angle'] = 0.0
    df['fe__vel_convergence'] = 0.0
    df['fe__def_coverage_density'] = 0.0
    
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
    CRITICAL: Add temporal derivative features (jerk, angular velocity, etc.)
    This function is from v10_test1_preprocessing.py and MUST be included!
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
    """
    Enhanced feature engineering with ALL v2 features
    CRITICAL: Must match v10_test1_preprocessing.py exactly!
    """
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
    
    # Apply all feature tiers (CRITICAL: Must include add_temporal_derivative_features!)
    df = compute_physics_features(df)
    df = compute_interaction_features(df)
    df = compute_context_features(df)
    df = create_zone_coverage_features(df)
    df = add_temporal_derivative_features(df)  # CRITICAL: This was in preprocessing!
    
    return df

def create_categorical_features(df):
    df = df.copy()
    play_direction = df['play_direction'] if 'play_direction' in df.columns else 'right'
    df['play_direction_left'] = (play_direction.str.lower() == 'left').astype(float)
    return df

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

# ============================================================================
# TTA FUNCTIONS
# ============================================================================

def flip_play(df):
    """Horizontal flip augmentation"""
    df_flipped = df.copy()
    
    if 'y' in df_flipped.columns:
        df_flipped['y'] = FIELD_Y_MAX - df_flipped['y']
    if 'ball_land_y' in df_flipped.columns:
        df_flipped['ball_land_y'] = FIELD_Y_MAX - df_flipped['ball_land_y']
    
    if 'dir' in df_flipped.columns:
        df_flipped['dir'] = (180 - df_flipped['dir']) % 360
    if 'o' in df_flipped.columns:
        df_flipped['o'] = (180 - df_flipped['o']) % 360
    
    if 'play_direction' in df_flipped.columns:
        play_dir = df_flipped['play_direction'].iloc[0]
        df_flipped['play_direction'] = 'left' if play_dir == 'right' else 'right'
    
    return df_flipped

def rotate_play(df, angle_degrees):
    """Rotation augmentation"""
    df_rotated = df.copy()
    
    center_x = FIELD_X_MAX / 2.0
    center_y = FIELD_Y_MAX / 2.0
    
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    if 'x' in df_rotated.columns and 'y' in df_rotated.columns:
        x_centered = df_rotated['x'] - center_x
        y_centered = df_rotated['y'] - center_y
        df_rotated['x'] = x_centered * cos_a - y_centered * sin_a + center_x
        df_rotated['y'] = x_centered * sin_a + y_centered * cos_a + center_y
    
    if 'ball_land_x' in df_rotated.columns and 'ball_land_y' in df_rotated.columns:
        x_centered = df_rotated['ball_land_x'] - center_x
        y_centered = df_rotated['ball_land_y'] - center_y
        df_rotated['ball_land_x'] = x_centered * cos_a - y_centered * sin_a + center_x
        df_rotated['ball_land_y'] = x_centered * sin_a + y_centered * cos_a + center_y
    
    if 'dir' in df_rotated.columns:
        df_rotated['dir'] = (df_rotated['dir'] + angle_degrees) % 360
    if 'o' in df_rotated.columns:
        df_rotated['o'] = (df_rotated['o'] + angle_degrees) % 360
    
    return df_rotated

def unrotate_predictions(predictions, angle_degrees):
    """Un-rotate predictions back to original coordinate system"""
    center_x = FIELD_X_MAX / 2.0
    center_y = FIELD_Y_MAX / 2.0
    
    angle_rad = np.deg2rad(-angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    unrotated = predictions.copy()
    x_centered = predictions[..., 0] - center_x
    y_centered = predictions[..., 1] - center_y
    
    unrotated[..., 0] = x_centered * cos_a - y_centered * sin_a + center_x
    unrotated[..., 1] = x_centered * sin_a + y_centered * cos_a + center_y
    
    return unrotated

# ============================================================================
# PREPROCESSING HELPERS
# ============================================================================

def transform_inputs(df, feature_cols):
    """Transform inputs using global preprocessor params"""
    global PREPROCESSOR_PARAMS
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    position_features = PREPROCESSOR_PARAMS['position_features']
    other_features = PREPROCESSOR_PARAMS['other_features']
    
    position_mean = np.array(PREPROCESSOR_PARAMS['position_mean'])
    position_scale = np.array(PREPROCESSOR_PARAMS['position_scale'])
    
    transformed_parts = []
    
    if position_features:
        position_data = df[position_features].fillna(0).values
        position_transformed = (position_data - position_mean) / (position_scale + 1e-8)
        transformed_parts.append(position_transformed)
    
    if other_features and PREPROCESSOR_PARAMS['other_mean'] is not None:
        other_mean = np.array(PREPROCESSOR_PARAMS['other_mean'])
        other_scale = np.array(PREPROCESSOR_PARAMS['other_scale'])
        other_data = df[other_features].fillna(0).values
        other_transformed = (other_data - other_mean) / (other_scale + 1e-8)
        other_transformed = np.clip(other_transformed, -10, 10)
        transformed_parts.append(other_transformed)
    
    all_numeric_features = position_features + other_features
    categorical_features = [f for f in feature_cols if f not in all_numeric_features]
    
    if categorical_features:
        cat_data = df[categorical_features].fillna(0).values
        transformed_parts.append(cat_data)
    
    return np.hstack(transformed_parts)

def inverse_transform_positions(predictions):
    """Un-scale predictions back to yards"""
    global PREPROCESSOR_PARAMS
    position_mean = np.array(PREPROCESSOR_PARAMS['position_mean'])
    position_scale = np.array(PREPROCESSOR_PARAMS['position_scale'])
    return predictions * (position_scale + 1e-8) + position_mean

# ============================================================================
# POST-PROCESSING
# ============================================================================

def apply_savgol_smoothing(trajectories, num_frames):
    """Apply Savitzky-Golay smoothing filter"""
    if num_frames < SAVGOL_WINDOW or not APPLY_SMOOTHING:
        return trajectories
    
    smoothed = trajectories.copy()
    for player_idx in range(trajectories.shape[0]):
        for coord_idx in range(2):
            smoothed[player_idx, :, coord_idx] = savgol_filter(
                trajectories[player_idx, :, coord_idx],
                window_length=SAVGOL_WINDOW,
                polyorder=SAVGOL_ORDER,
                mode='nearest'
            )
    return smoothed

def enforce_max_speed(trajectories, num_frames):
    """Enforce maximum speed constraint"""
    if not APPLY_SPEED_CAP or num_frames < 2:
        return trajectories
    
    constrained = trajectories.copy()
    max_distance_per_frame = MAX_SPEED_YD_PER_SEC / FRAME_RATE
    
    for player_idx in range(trajectories.shape[0]):
        for frame_idx in range(1, num_frames):
            prev_pos = constrained[player_idx, frame_idx - 1]
            curr_pos = constrained[player_idx, frame_idx]
            
            delta = curr_pos - prev_pos
            distance = np.linalg.norm(delta)
            
            if distance > max_distance_per_frame:
                scale_factor = max_distance_per_frame / distance
                constrained[player_idx, frame_idx] = prev_pos + (delta * scale_factor)
    
    return constrained

# ============================================================================
# MODEL LOADING WITH ADAPTIVE WEIGHTING
# ============================================================================

def load_models():
    """
    Load models with adaptive IQR-based filtering and weighted averaging
    Uses α=2.5 penalty and weight clamping
    """
    global MODELS_AND_WEIGHTS, PREPROCESSOR_PARAMS, DEVICE
    
    print("[INFO] Loading models and preprocessor for v10 ReduceOnPlateau...")
    print(f"[INFO] Expected seeds: {ENSEMBLE_SEEDS}")
    print(f"[INFO] Adaptive weighting: α={ALPHA}, min_w={MIN_WEIGHT}, max_w={MAX_WEIGHT}")
    
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    
    PREPROCESSOR_PARAMS = torch.load(PREPROCESSOR_PATH, map_location=DEVICE, weights_only=False)
    feature_cols = PREPROCESSOR_PARAMS['feature_cols']
    input_dim = len(feature_cols)
    
    print(f"[INFO] Preprocessor loaded: {input_dim} features")
    print(f"[INFO] Architecture: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}, LOOKBACK={LOOKBACK}")
    
    models_with_rmse = []
    
    for seed in ENSEMBLE_SEEDS:
        model_path = MODELS_DIR / f"model_seed{seed}.pth"
        
        if not model_path.exists():
            print(f"[WARNING] Model not found: {model_path}, skipping...")
            continue
        
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        val_rmse = checkpoint.get('val_rmse')
        if val_rmse is None:
            print(f"[WARNING] Model {seed} missing 'val_rmse', skipping...")
            continue
        
        if checkpoint['input_dim'] != input_dim:
            print(f"[WARNING] Model {seed} dimension mismatch (expected {input_dim}, got {checkpoint['input_dim']}), skipping...")
            continue
        
        # Verify architecture parameters match
        expected_d_model = checkpoint.get('d_model', D_MODEL)
        expected_nhead = checkpoint.get('nhead', N_HEADS)
        expected_n_layers = checkpoint.get('n_layers', N_LAYERS)
        
        if expected_d_model != D_MODEL or expected_nhead != N_HEADS or expected_n_layers != N_LAYERS:
            print(f"[WARNING] Model {seed} architecture mismatch, skipping...")
            continue
        
        model = HybridModel(
            input_dim=checkpoint['input_dim'],
            d_model=checkpoint['d_model'],
            gnn_dim=128,
            decoder_hidden=128,
            nhead=checkpoint['nhead'],
            num_encoder_layers=checkpoint['n_layers']
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models_with_rmse.append((model, val_rmse))
        print(f"[INFO] ✅ Loaded model seed {seed} (Val RMSE: {val_rmse:.4f})")
    
    if len(models_with_rmse) == 0:
        raise RuntimeError(f"No models found! Check MODELS_DIR: {MODELS_DIR}")
    
    # ============================================================================
    # ADAPTIVE IQR-BASED OUTLIER FILTERING
    # ============================================================================
    rmses = np.array([rmse for _, rmse in models_with_rmse], dtype=float)
    
    if len(rmses) > 1:
        q1, q3 = np.percentile(rmses, [25, 75])
        iqr = max(q3 - q1, 1e-6)
        upper_cut = q3 + 1.5 * iqr
        keep_mask = rmses <= upper_cut
        
        models_with_rmse = [m for m, k in zip(models_with_rmse, keep_mask) if k]
        rmses = rmses[keep_mask]
        
        print(f"[INFO] Adaptive IQR trim kept {len(models_with_rmse)} models (removed {(~keep_mask).sum()})")
    
    if len(models_with_rmse) == 0:
        raise RuntimeError("All models filtered out by IQR!")
    
    # ============================================================================
    # WEIGHTED AVERAGING WITH α PENALTY
    # ============================================================================
    raw_w = rmses ** (-ALPHA)
    w = raw_w / raw_w.sum()
    
    # Clamp weights
    w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
    w = w / w.sum()  # Re-normalize
    
    MODELS_AND_WEIGHTS = [(models_with_rmse[i][0], float(w[i])) for i in range(len(w))]
    
    print(f"[INFO] α = {ALPHA}, weights: {[round(float(x), 3) for x in w]}")
    print(f"[INFO] ✅ Loaded {len(MODELS_AND_WEIGHTS)} models successfully")
    print(f"[INFO] Device: {DEVICE}")
    
    if APPLY_TTA:
        print("[INFO] 🆕 Test-Time Augmentation (TTA) - Horizontal Flip enabled")
    if APPLY_ROTATION_TTA:
        print(f"[INFO] 🆕 Test-Time Augmentation (TTA) - Rotation enabled. Angles: {ROTATION_ANGLES}°")
    if APPLY_SMOOTHING:
        print(f"[INFO] 🆕 Savitzky-Golay smoothing enabled (window={SAVGOL_WINDOW}, order={SAVGOL_ORDER})")
    if APPLY_SPEED_CAP:
        print(f"[INFO] 🆕 Max speed constraint enabled ({MAX_SPEED_YD_PER_SEC} yd/s)")

# ============================================================================
# PLAY PROCESSING
# ============================================================================

def process_test_play(play_input, passer_row, feature_cols):
    """Prepare a single play for model inference"""
    
    if 'player_to_predict' not in play_input.columns:
        play_input = identify_players_to_predict(play_input)
    
    players = play_input[play_input['player_to_predict'] == True]['nfl_id'].unique()
    if len(players) == 0:
        players = play_input['nfl_id'].unique()
    
    # Throw features (8-dim from v10)
    throw_features = np.array([
        passer_row.get('x', 50.0) / 120.0,
        passer_row.get('y', 26.65) / 53.3,
        passer_row.get('ball_land_x', 50.0) / 120.0,
        passer_row.get('ball_land_y', 26.65) / 53.3,
        np.cos(np.deg2rad(passer_row.get('dir', 0.0))),
        np.sin(np.deg2rad(passer_row.get('dir', 0.0))),
        0.0, 0.0
    ], dtype=np.float32)
    
    input_sequences, valid_players, player_positions = [], [], []
    
    for nfl_id in players:
        player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
        
        # Padding logic (v10 uses LOOKBACK=25)
        if len(player_input) >= LOOKBACK:
            player_input = player_input.tail(LOOKBACK)
        else:
            pad_len = LOOKBACK - len(player_input)
            pad_df = pd.DataFrame([player_input.iloc[0]] * pad_len)
            player_input = pd.concat([pad_df, player_input], ignore_index=True)
        
        features = transform_inputs(player_input[feature_cols].fillna(0), feature_cols)
        input_sequences.append(features)
        valid_players.append(nfl_id)
        player_positions.append(player_input[['x', 'y']].iloc[-1].values)
    
    if len(valid_players) == 0:
        return None
    
    num_players = len(valid_players)
    player_positions = np.array(player_positions)
    
    # Build graph edges
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
    
    return {
        'input_sequence': torch.tensor(np.array(input_sequences), dtype=torch.float32).unsqueeze(0),
        'throw_features': torch.tensor(throw_features, dtype=torch.float32).unsqueeze(0),
        'edge_index': edge_index.unsqueeze(0),
        'edge_weights': edge_weights.unsqueeze(0),
        'edge_mask': torch.ones(1, len(edges), dtype=torch.bool),
        'valid_players': valid_players,
        'num_players': len(valid_players),
        'last_known_positions': {nfl_id: pos for nfl_id, pos in zip(valid_players, player_positions)}
    }

# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    """
    Main prediction function with:
    - Adaptive weighted averaging
    - TTA (flip + rotation)
    - Post-processing (smoothing + speed cap)
    - Fixed fallback logic
    """
    global MODELS_AND_WEIGHTS, PREPROCESSOR_PARAMS
    
    if MODELS_AND_WEIGHTS is None:
        load_models()
    
    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()
    
    # Apply v10 feature engineering to original (MUST include add_temporal_derivative_features!)
    test_input_orig_pd = engineer_features(test_input_pd.copy())
    test_input_orig_pd = create_categorical_features(test_input_orig_pd)
    
    # Prepare augmented versions
    augmented_versions = []
    
    # 1. Horizontal flip
    if APPLY_TTA:
        test_input_flipped_pd = flip_play(test_input_pd.copy())
        test_input_flipped_pd = engineer_features(test_input_flipped_pd)
        test_input_flipped_pd = create_categorical_features(test_input_flipped_pd)
        augmented_versions.append(('flip', test_input_flipped_pd))
    
    # 2. Rotations
    if APPLY_ROTATION_TTA:
        for angle in ROTATION_ANGLES:
            test_input_rotated_pd = rotate_play(test_input_pd.copy(), angle)
            test_input_rotated_pd = engineer_features(test_input_rotated_pd)
            test_input_rotated_pd = create_categorical_features(test_input_rotated_pd)
            augmented_versions.append(('rotate', test_input_rotated_pd, angle))
    
    feature_cols = PREPROCESSOR_PARAMS['feature_cols']
    all_predictions = []
    
    # Process each play
    for (game_id, play_id), play_group_orig in test_input_orig_pd.groupby(['game_id', 'play_id']):
        test_rows = test_pd[
            (test_pd['game_id'] == game_id) & 
            (test_pd['play_id'] == play_id)
        ].copy()
        
        if len(test_rows) == 0:
            continue
        
        # Get passer info
        passer_mask_orig = play_group_orig['player_role'] == 'Passer' if 'player_role' in play_group_orig.columns else pd.Series([False] * len(play_group_orig))
        passer_row_orig = play_group_orig[passer_mask_orig].iloc[0] if passer_mask_orig.any() else play_group_orig.iloc[0]
        
        # Process original play
        play_data_orig = process_test_play(play_group_orig, passer_row_orig, feature_cols)
        
        # Store last known positions for fallback
        last_known_positions = {}
        if play_data_orig:
            last_known_positions = play_data_orig['last_known_positions']
        else:
            for nfl_id, group in play_group_orig.groupby('nfl_id'):
                last_pos = group.iloc[-1]
                last_known_positions[nfl_id] = np.array([last_pos['x'], last_pos['y']])
        
        if play_data_orig is None:
            all_predictions.append(test_rows[['x', 'y']])
            continue
        
        num_frames = int(play_group_orig['num_frames_output'].iloc[0])
        
        # Process augmented versions
        augmented_play_data = []
        for aug_info in augmented_versions:
            aug_type = aug_info[0]
            aug_df = aug_info[1]
            
            play_group_aug = aug_df.loc[play_group_orig.index]
            passer_mask_aug = play_group_aug['player_role'] == 'Passer' if 'player_role' in play_group_aug.columns else pd.Series([False] * len(play_group_aug))
            passer_row_aug = play_group_aug[passer_mask_aug].iloc[0] if passer_mask_aug.any() else play_group_aug.iloc[0]
            play_data_aug = process_test_play(play_group_aug, passer_row_aug, feature_cols)
            
            if play_data_aug is not None:
                if aug_type == 'flip':
                    augmented_play_data.append(('flip', play_data_aug, None))
                elif aug_type == 'rotate':
                    angle = aug_info[2]
                    augmented_play_data.append(('rotate', play_data_aug, angle))
        
        # Run weighted ensemble with TTA
        weighted_preds_list = []
        
        with torch.no_grad():
            for model, weight in MODELS_AND_WEIGHTS:
                # Original prediction
                preds_orig_scaled, _ = model(
                    play_data_orig['input_sequence'].to(DEVICE),
                    play_data_orig['throw_features'].to(DEVICE),
                    play_data_orig['edge_index'].to(DEVICE),
                    play_data_orig['edge_weights'].to(DEVICE),
                    play_data_orig['edge_mask'].to(DEVICE),
                    num_frames
                )
                preds_orig_yards = inverse_transform_positions(
                    preds_orig_scaled.squeeze(0).cpu().numpy()
                )
                
                # Collect augmented predictions
                aug_preds_list = [preds_orig_yards]
                
                # Process all augmentations
                for aug_type, play_data_aug, extra_info in augmented_play_data:
                    preds_aug_scaled, _ = model(
                        play_data_aug['input_sequence'].to(DEVICE),
                        play_data_aug['throw_features'].to(DEVICE),
                        play_data_aug['edge_index'].to(DEVICE),
                        play_data_aug['edge_weights'].to(DEVICE),
                        play_data_aug['edge_mask'].to(DEVICE),
                        num_frames
                    )
                    preds_aug_yards = inverse_transform_positions(
                        preds_aug_scaled.squeeze(0).cpu().numpy()
                    )
                    
                    # Un-augment predictions
                    if aug_type == 'flip':
                        preds_aug_yards[..., 1] = FIELD_Y_MAX - preds_aug_yards[..., 1]
                    elif aug_type == 'rotate':
                        angle = extra_info
                        preds_aug_yards = unrotate_predictions(preds_aug_yards, angle)
                    
                    aug_preds_list.append(preds_aug_yards)
                
                # Average all TTA versions for this model
                final_preds_yards = np.mean(aug_preds_list, axis=0)
                
                # Add weighted prediction
                weighted_preds_list.append(final_preds_yards * weight)
        
        # Combine all models (already weighted)
        avg_preds_yards = np.sum(weighted_preds_list, axis=0)
        avg_preds_yards = avg_preds_yards.reshape(play_data_orig['num_players'], num_frames, 2)
        
        # Post-processing
        avg_preds_yards = apply_savgol_smoothing(avg_preds_yards, num_frames)
        avg_preds_yards = enforce_max_speed(avg_preds_yards, num_frames)
        
        # Clip to field boundaries
        avg_preds_yards[:, :, 0] = np.clip(avg_preds_yards[:, :, 0], 0, FIELD_X_MAX)
        avg_preds_yards[:, :, 1] = np.clip(avg_preds_yards[:, :, 1], 0, FIELD_Y_MAX)
        
        # Fill submission with fallback logic
        valid_players = play_data_orig['valid_players']
        
        for idx, row in test_rows.iterrows():
            nfl_id = row['nfl_id']
            frame_id = row['frame_id']
            
            try:
                player_idx = valid_players.index(nfl_id)
                frame_idx = frame_id - 1
                
                if 0 <= frame_idx < num_frames:
                    test_rows.at[idx, 'x'] = avg_preds_yards[player_idx, frame_idx, 0]
                    test_rows.at[idx, 'y'] = avg_preds_yards[player_idx, frame_idx, 1]
                else:
                    # Use last frame
                    test_rows.at[idx, 'x'] = avg_preds_yards[player_idx, -1, 0]
                    test_rows.at[idx, 'y'] = avg_preds_yards[player_idx, -1, 1]
            
            except (ValueError, IndexError):
                # Fallback to last known position
                if nfl_id in last_known_positions:
                    test_rows.at[idx, 'x'] = last_known_positions[nfl_id][0]
                    test_rows.at[idx, 'y'] = last_known_positions[nfl_id][1]
                else:
                    # Absolute fallback
                    test_rows.at[idx, 'x'] = 50.0
                    test_rows.at[idx, 'y'] = 26.65
        
        all_predictions.append(test_rows[['x', 'y']])
    
    if len(all_predictions) == 0:
        predictions = pd.DataFrame({'x': [50.0] * len(test_pd), 'y': [26.65] * len(test_pd)})
    else:
        predictions = pd.concat(all_predictions, ignore_index=True)
    
    predictions_pl = pl.from_pandas(predictions)
    
    assert len(predictions_pl) == len(test)
    return predictions_pl

# ============================================================================
# SERVER SETUP
# ============================================================================

print("="*80)
print("NFL GNN-Transformer Inference v10 ReduceOnPlateau")
print("="*80)
print(f"Models directory: {MODELS_DIR}")
print(f"Preprocessor: {PREPROCESSOR_PATH}")
print(f"Expected seeds: {ENSEMBLE_SEEDS}")
print(f"LOOKBACK: {LOOKBACK}")
print(f"Model architecture: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}")
print(f"Adaptive weighting: α={ALPHA}, min_w={MIN_WEIGHT}, max_w={MAX_WEIGHT}")
print("✅ CRITICAL: Using add_temporal_derivative_features from v10_test1_preprocessing.py")
print("="*80)

inference_server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Serving predictions...")
    inference_server.serve()
else:
    print("Running local gateway for debugging...")
    inference_server.run_local_gateway(
        ('/kaggle/input/nfl-big-data-bowl-2026-prediction/',)
    )
    print("Local gateway run finished.")

