"""
NFL Training Pipeline - v10 (Feature Test)

This script trains models using preprocessed data with advanced v2 features.
Run AFTER preprocessing is complete.

Usage:
    python nfl_training_v10.py
"""

import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
import csv
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
PROCESSED_DATA_DIR = Path("/content/drive/MyDrive/NFL Big Data Bowl Prediction/test1/working/processed_data")
MODELS_DIR = Path("/content/drive/MyDrive/NFL Big Data Bowl Prediction/test1/working/models")
LOGS_DIR = Path("/content/drive/MyDrive/NFL Big Data Bowl Prediction/test1/working/logs")

# Training seeds
ENSEMBLE_SEEDS = [42]

# Training config
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGNIFICANT_IMPROVEMENT_PATIENCE = 20
SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.001

# Model architecture
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3

# Data augmentation
NOISE_LEVEL = 0.03

# From preprocessing
LOOKBACK = 25

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class TrainingLogger:
    """Logs training metrics to CSV files"""
    def __init__(self, log_dir, seed):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.csv_path = self.log_dir / f"training_log_seed_{seed}.csv"

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'seed', 'epoch',
                'train_loss', 'train_rmse', 'train_ade', 'train_fde', 'train_player_loss', 'train_ball_loss',
                'val_loss', 'val_rmse', 'val_ade', 'val_fde', 'val_player_loss', 'val_ball_loss',
                'learning_rate', 'epoch_time_sec', 'is_best_epoch', 'frames_trained'
            ])
            f.flush()
            os.fsync(f.fileno())

        print(f"[INFO] 📝 Logging to: {self.csv_path}")

    def log_epoch(self, epoch, train_metrics, val_metrics, lr, is_best=False):
        """Log metrics for one epoch"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, self.seed, epoch,
                train_metrics['loss'], train_metrics['rmse'], train_metrics['ade'], train_metrics['fde'],
                train_metrics['player_loss'], train_metrics['ball_loss'],
                val_metrics['loss'], val_metrics['rmse'], val_metrics['ade'], val_metrics['fde'],
                val_metrics['player_loss'], val_metrics['ball_loss'],
                lr, train_metrics['time'], is_best, train_metrics.get('frames_trained', LOOKBACK)
            ])
            f.flush()
            os.fsync(f.fileno())

    def save_summary(self, best_epoch, best_rmse, total_epochs, total_time):
        """Save training summary"""
        summary_path = self.log_dir / f"summary_seed_{self.seed}.json"

        summary = {
            'seed': self.seed,
            'best_epoch': best_epoch,
            'best_val_rmse': float(best_rmse),
            'total_epochs': total_epochs,
            'total_training_time_sec': float(total_time),
            'csv_log': str(self.csv_path)
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] 📄 Summary saved: {summary_path}")

def create_combined_log(logs_dir, ensemble_results):
    """Create a combined CSV with all models' best results"""
    combined_path = Path(logs_dir) / "combined_best_results.csv"

    with open(combined_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'best_val_rmse', 'training_time_min', 'model_path'])

        for result in ensemble_results:
            writer.writerow([
                result['seed'],
                result['best_val_rmse'],
                result['training_time_minutes'],
                result['model_path']
            ])
            f.flush()
            os.fsync(f.fileno())

    print(f"[INFO] 📊 Combined results saved: {combined_path}")

# ============================================================================
# DATASET
# ============================================================================

class FastNFLTrajectoryDataset(Dataset):
    """Loads consolidated weekly files"""
    def __init__(self, data_dir):
        super().__init__()
        self.week_files = sorted(list(Path(data_dir).glob('week_*.pt')))

        if not self.week_files:
            raise FileNotFoundError(f"No 'week_XX.pt' files in {data_dir}. Run preprocessing first!")

        self.all_plays = []
        print(f"[INFO] Loading data from {data_dir}...")
        for f in tqdm(self.week_files, desc="Loading data files"):
            self.all_plays.extend(torch.load(f))

        self.is_train = 'train' in str(data_dir).lower()

        print(f"[INFO] Dataset loaded: {len(self.all_plays)} plays from {len(self.week_files)} files")
        if self.is_train:
            print(f"[INFO] 🎲 Data augmentation enabled (noise level: {NOISE_LEVEL})")

    def __len__(self):
        return len(self.all_plays)

    def __getitem__(self, idx):
        data = self.all_plays[idx]

        if self.is_train:
            noise = torch.randn_like(data['input_sequence']) * NOISE_LEVEL
            data['input_sequence'] = data['input_sequence'] + noise

        return {
            'input_sequence': data['input_sequence'],
            'target_sequence': data['target_sequence'],
            'throw_features': data['throw_features'],
            'gt_ball_land_xy': data['gt_ball_land_xy'],
            'num_players': data['num_players'],
            'edge_index': data['edge_index'],
            'edge_weights': data['edge_weights'],
            'player_mask': data['player_mask']
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    if len(batch) == 0:
        raise ValueError("Empty batch")

    batch = [item for item in batch if item is not None and item['num_players'] > 0]
    if len(batch) == 0:
        return {'input_sequence': torch.empty(0), 'target_sequence': torch.empty(0),
                'throw_features': torch.empty(0), 'gt_ball_land_xy': torch.empty(0),
                'player_mask': torch.empty(0), 'edge_index': torch.empty(0),
                'edge_weights': torch.empty(0), 'edge_mask': torch.empty(0)}

    max_players = max(item['num_players'] for item in batch)
    max_target_frames = max(item['target_sequence'].shape[1] for item in batch)
    max_edges = max(item['edge_index'].shape[1] for item in batch)

    batch_size = len(batch)
    lookback_frames = batch[0]['input_sequence'].shape[1]
    input_features = batch[0]['input_sequence'].shape[2]

    padded_input = torch.zeros(batch_size, max_players, lookback_frames, input_features)
    padded_target = torch.zeros(batch_size, max_players, max_target_frames, 2)
    player_mask = torch.zeros(batch_size, max_players, dtype=torch.bool)

    padded_edge_index = torch.zeros(batch_size, 2, max_edges, dtype=torch.long)
    padded_edge_weights = torch.zeros(batch_size, max_edges)
    edge_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)

    throw_features_list = []
    gt_ball_land_xy_list = []

    for i, item in enumerate(batch):
        num_players = item['num_players']
        num_edges = item['edge_index'].shape[1]

        padded_input[i, :num_players] = item['input_sequence']
        padded_target[i, :num_players, :item['target_sequence'].shape[1]] = item['target_sequence']
        player_mask[i, :num_players] = True

        padded_edge_index[i, :, :num_edges] = item['edge_index']
        padded_edge_weights[i, :num_edges] = item['edge_weights']
        edge_mask[i, :num_edges] = True

        throw_features_list.append(item['throw_features'])
        gt_ball_land_xy_list.append(item['gt_ball_land_xy'])

    return {
        'input_sequence': padded_input,
        'target_sequence': padded_target,
        'throw_features': torch.stack(throw_features_list),
        'gt_ball_land_xy': torch.stack(gt_ball_land_xy_list),
        'player_mask': player_mask,
        'edge_index': padded_edge_index,
        'edge_weights': padded_edge_weights,
        'edge_mask': edge_mask
    }

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class BallTrajectoryPredictor(nn.Module):
    """Predicts ball landing position"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.1)
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 3)
        )
    
    def forward(self, throw_features):
        feat = self.encoder(throw_features).unsqueeze(1)
        _, hidden = self.rnn(feat)
        return self.output(hidden[-1])

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
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
    """Graph convolution with attention"""
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
            src_feats, tgt_feats = node_features[b, src_nodes], node_features[b, tgt_nodes]
            
            concat = torch.cat([tgt_feats, src_feats], dim=-1)
            attn_scores = torch.sigmoid(self.attention(concat))
            messages = attn_scores * weights.unsqueeze(-1) * transformed[b, src_nodes]
            
            aggregated[b].index_add_(0, tgt_nodes, messages.to(aggregated[b].dtype))
        
        return transformed + aggregated

class HybridModel(nn.Module):
    """Main trajectory prediction model"""
    def __init__(self, input_dim, d_model=D_MODEL, gnn_dim=128,
                 decoder_hidden=128, nhead=N_HEADS, num_encoder_layers=N_LAYERS):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        self.ball_predictor = BallTrajectoryPredictor()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=LOOKBACK)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.gnn1 = OptimizedGraphConvLayer(d_model, gnn_dim)
        self.gnn2 = OptimizedGraphConvLayer(gnn_dim, gnn_dim)
        
        self.decoder_gru = nn.GRU(
            input_size=gnn_dim + 2 + 3, 
            hidden_size=decoder_hidden,
            num_layers=2, 
            batch_first=True, 
            dropout=0.1
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(decoder_hidden, 64), 
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input_sequence, throw_features, edge_index, edge_weights, edge_mask, num_steps):
        batch_size, num_players, seq_len, features = input_sequence.shape
        
        assert features == self.input_dim, f"Model input_dim ({self.input_dim}) != data features ({features})"
        assert seq_len == LOOKBACK, f"Model LOOKBACK ({LOOKBACK}) != data seq_len ({seq_len})"
        
        # Predict ball trajectory
        pcp = self.ball_predictor(throw_features)
        
        # Process player sequences
        x = input_sequence.reshape(batch_size * num_players, seq_len, features)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Extract context and apply GNN
        context = x[:, -1, :].reshape(batch_size, num_players, self.d_model)
        context = F.relu(self.gnn1(context, edge_index, edge_weights, edge_mask))
        context = self.gnn2(context, edge_index, edge_weights, edge_mask)
        
        # Prepare for decoding
        pcp_expanded = pcp[:, :2].unsqueeze(1).expand(-1, num_players, -1)
        context_flat = context.reshape(batch_size * num_players, -1)
        pcp_flat = pcp_expanded.reshape(batch_size * num_players, 2)
        pcp_conf = pcp[:, 2:3].unsqueeze(1).expand(-1, num_players, -1).reshape(-1, 1)
        
        # Autoregressive decoding
        predictions = []
        current_pos = pcp_flat.unsqueeze(1)
        hidden = None
        
        for _ in range(num_steps):
            decoder_input = torch.cat([
                current_pos, 
                context_flat.unsqueeze(1),
                pcp_flat.unsqueeze(1), 
                pcp_conf.unsqueeze(1)
            ], dim=-1)
            
            output, hidden = self.decoder_gru(decoder_input, hidden)
            next_pos = self.output_layer(output)
            predictions.append(next_pos)
            current_pos = next_pos
        
        predictions = torch.cat(predictions, dim=1)
        return predictions.reshape(batch_size, num_players, num_steps, 2), pcp

# ============================================================================
# LOSS FUNCTION
# ============================================================================

class CleanPositionLoss(nn.Module):
    """MSE loss with metrics in yards"""
    def __init__(self, preprocessor_params, ball_weight=0.1):
        super().__init__()
        self.ball_weight = ball_weight
        self.player_loss_fn = nn.MSELoss()
        self.ball_loss_fn = nn.MSELoss()

        self.position_mean = np.array(preprocessor_params['position_mean'])
        self.position_scale = np.array(preprocessor_params['position_scale'])

    def inverse_transform(self, targets):
        """Converts scaled data back to yards"""
        return targets * (self.position_scale + 1e-8) + self.position_mean

    def forward(self, predictions, targets, pcp, gt_ball_land_xy, player_mask):
        # Calculate losses
        batch_size, num_players, num_frames, _ = predictions.shape
        
        mask_expanded = player_mask.unsqueeze(-1).unsqueeze(-1)
        masked_preds = predictions * mask_expanded
        masked_targets = targets * mask_expanded
        
        player_loss = self.player_loss_fn(masked_preds, masked_targets)
        ball_loss = self.ball_loss_fn(pcp[:, :2], gt_ball_land_xy)
        total_loss = player_loss + (self.ball_weight * ball_loss)

        # Calculate metrics in yards
        with torch.no_grad():
            rmse_list, ade_list, fde_list = [], [], []

            for b in range(batch_size):
                valid_mask = player_mask[b]
                if not valid_mask.any():
                    continue

                num_valid = valid_mask.sum().item()
                num_frames = predictions.shape[2]
                
                if num_frames == 0:
                    continue

                pred_b = predictions[b, valid_mask].reshape(-1, 2).cpu().numpy()
                target_b = targets[b, valid_mask].reshape(-1, 2).cpu().numpy()

                pred_yards = self.inverse_transform(pred_b).reshape(num_valid, num_frames, 2)
                target_yards = self.inverse_transform(target_b).reshape(num_valid, num_frames, 2)

                errors = np.sqrt(((pred_yards - target_yards) ** 2).sum(axis=-1))

                rmse_list.append(np.sqrt((errors ** 2).mean()))
                ade_list.append(errors.mean())
                fde_list.append(errors[:, -1].mean())

            rmse_yards = np.mean(rmse_list) if rmse_list else 0.0
            ade_yards = np.mean(ade_list) if ade_list else 0.0
            fde_yards = np.mean(fde_list) if fde_list else 0.0

        return total_loss, {
            'loss': total_loss.item(),
            'player_loss': player_loss.item(),
            'ball_loss': ball_loss.item(),
            'rmse': rmse_yards,
            'ade': ade_yards,
            'fde': fde_yards
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = total_rmse = total_ade = total_fde = 0
    total_player_loss = total_ball_loss = 0
    num_batches = 0

    epoch_start = time.time()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        if batch['input_sequence'].numel() == 0:
            continue

        input_seq = batch['input_sequence'].to(device, non_blocking=True)
        target_seq = batch['target_sequence'].to(device, non_blocking=True)
        throw_feat = batch['throw_features'].to(device, non_blocking=True)
        gt_ball_xy = batch['gt_ball_land_xy'].to(device, non_blocking=True)
        player_mask = batch['player_mask'].to(device, non_blocking=True)
        edge_index = batch['edge_index'].to(device, non_blocking=True)
        edge_weights = batch['edge_weights'].to(device, non_blocking=True)
        edge_mask = batch['edge_mask'].to(device, non_blocking=True)

        num_frames = target_seq.shape[2]
        if num_frames == 0:
            continue

        if USE_MIXED_PRECISION and scaler:
            with autocast('cuda'):
                predictions, pred_pcp = model(input_seq, throw_feat, edge_index,
                                              edge_weights, edge_mask, num_frames)
                loss, loss_dict = criterion(predictions, target_seq, pred_pcp, gt_ball_xy, player_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            predictions, pred_pcp = model(input_seq, throw_feat, edge_index,
                                          edge_weights, edge_mask, num_frames)
            loss, loss_dict = criterion(predictions, target_seq, pred_pcp, gt_ball_xy, player_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict['loss']
        total_player_loss += loss_dict['player_loss']
        total_ball_loss += loss_dict['ball_loss']
        total_rmse += loss_dict['rmse']
        total_ade += loss_dict['ade']
        total_fde += loss_dict['fde']
        num_batches += 1

        progress_bar.set_postfix({
            'RMSE': f'{loss_dict["rmse"]:.2f}',
            'ball': f'{loss_dict["ball_loss"]:.4f}'
        })

    epoch_time = time.time() - epoch_start

    if num_batches == 0:
        return {'loss': 0.0, 'rmse': 0.0, 'ade': 0.0, 'fde': 0.0, 'time': epoch_time,
                'player_loss': 0.0, 'ball_loss': 0.0, 'frames_trained': num_frames}

    return {
        'loss': total_loss / num_batches,
        'player_loss': total_player_loss / num_batches,
        'ball_loss': total_ball_loss / num_batches,
        'rmse': total_rmse / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches,
        'time': epoch_time,
        'frames_trained': num_frames
    }

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = total_rmse = total_ade = total_fde = 0
    total_player_loss = total_ball_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            if batch['input_sequence'].numel() == 0:
                continue

            input_seq = batch['input_sequence'].to(device, non_blocking=True)
            target_seq = batch['target_sequence'].to(device, non_blocking=True)
            throw_feat = batch['throw_features'].to(device, non_blocking=True)
            gt_ball_xy = batch['gt_ball_land_xy'].to(device, non_blocking=True)
            player_mask = batch['player_mask'].to(device, non_blocking=True)
            edge_index = batch['edge_index'].to(device, non_blocking=True)
            edge_weights = batch['edge_weights'].to(device, non_blocking=True)
            edge_mask = batch['edge_mask'].to(device, non_blocking=True)

            max_frames = target_seq.shape[2]
            if max_frames == 0:
                continue

            if USE_MIXED_PRECISION:
                with autocast('cuda'):
                    predictions, pred_pcp = model(input_seq, throw_feat, edge_index,
                                                 edge_weights, edge_mask, max_frames)
                    loss, loss_dict = criterion(predictions, target_seq, pred_pcp, gt_ball_xy, player_mask)
            else:
                predictions, pred_pcp = model(input_seq, throw_feat, edge_index,
                                             edge_weights, edge_mask, max_frames)
                loss, loss_dict = criterion(predictions, target_seq, pred_pcp, gt_ball_xy, player_mask)

            total_loss += loss_dict['loss']
            total_player_loss += loss_dict['player_loss']
            total_ball_loss += loss_dict['ball_loss']
            total_rmse += loss_dict['rmse']
            total_ade += loss_dict['ade']
            total_fde += loss_dict['fde']
            num_batches += 1

    if num_batches == 0:
        return {'loss': float('inf'), 'rmse': float('inf'), 'ade': float('inf'),
                'fde': float('inf'), 'player_loss': float('inf'), 'ball_loss': float('inf')}

    return {
        'loss': total_loss / num_batches,
        'player_loss': total_player_loss / num_batches,
        'ball_loss': total_ball_loss / num_batches,
        'rmse': total_rmse / num_batches,
        'ade': total_ade / num_batches,
        'fde': total_fde / num_batches
    }

def train_single_model(seed, train_loader, val_loader, preprocessor_params, input_dim):
    """Train a single model with given seed"""
    print(f"\n{'='*80}")
    print(f"🎲 TRAINING MODEL WITH SEED {seed}")
    print(f"{'='*80}")

    logger = TrainingLogger(LOGS_DIR, seed)
    set_seed(seed)

    model = HybridModel(
        input_dim=input_dim,
        d_model=D_MODEL,
        gnn_dim=128,
        decoder_hidden=128,
        nhead=N_HEADS,
        num_encoder_layers=N_LAYERS
    ).to(DEVICE)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    criterion = CleanPositionLoss(preprocessor_params, ball_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler('cuda') if USE_MIXED_PRECISION else None

    best_val_rmse = float('inf')
    best_epoch = 0
    significant_counter = 0
    training_history = []

    training_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch, scaler)
        val_metrics = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        is_best = val_metrics['rmse'] < best_val_rmse

        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, is_best)

        print(f"\n{'─'*80}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  📈 Train | Loss: {train_metrics['loss']:.4f} | RMSE: {train_metrics['rmse']:.3f} | ADE: {train_metrics['ade']:.3f} | FDE: {train_metrics['fde']:.3f}")
        print(f"  📉 Val   | Loss: {val_metrics['loss']:.4f} | RMSE: {val_metrics['rmse']:.3f} | ADE: {val_metrics['ade']:.3f} | FDE: {val_metrics['fde']:.3f}")
        print(f"  🎯 Ball Loss - Train: {train_metrics['ball_loss']:.4f} | Val: {val_metrics['ball_loss']:.4f}")
        print(f"  ⏱️ Time: {train_metrics['time']:.1f}s | LR: {current_lr:.6f}")

        training_history.append({
            'epoch': epoch, 
            'train': train_metrics,
            'val': val_metrics, 
            'lr': current_lr
        })

        new_rmse = val_metrics['rmse']
        improved = new_rmse < best_val_rmse

        if improved:
            improvement = best_val_rmse - new_rmse if best_val_rmse != float('inf') else float('inf')
            best_val_rmse = new_rmse
            best_epoch = epoch

            if improvement >= SIGNIFICANT_IMPROVEMENT_THRESHOLD:
                significant_counter = 0
            else:
                significant_counter += 1

            checkpoint_path = MODELS_DIR / f"model_seed{seed}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rmse': best_val_rmse,
                'val_ade': val_metrics['ade'],
                'val_fde': val_metrics['fde'],
                'input_dim': input_dim,
                'd_model': D_MODEL,
                'nhead': N_HEADS,
                'n_layers': N_LAYERS,
                'training_history': training_history,
                'feature_cols': preprocessor_params['feature_cols'],
                'seed': seed,
                'preprocessor_params': preprocessor_params,
                'lookback': LOOKBACK
            }, checkpoint_path)

            print(f"  ✅ New best! RMSE: {best_val_rmse:.4f} (improved by {improvement:.4f})")
        else:
            significant_counter += 1
            print(f"  ⚠️ No improvement (best: {best_val_rmse:.4f})")

        print(f"  📊 Small-improvement: {significant_counter}/{SIGNIFICANT_IMPROVEMENT_PATIENCE}")
        print(f"{'─'*80}")

        if significant_counter >= SIGNIFICANT_IMPROVEMENT_PATIENCE:
            print(f"\n⛔ Early stopping (insufficient improvement) at epoch {epoch}")
            break

    total_training_time = time.time() - training_start_time

    logger.save_summary(best_epoch, best_val_rmse, epoch, total_training_time)

    print(f"\n✅ Model (seed {seed}) complete!")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best Val RMSE: {best_val_rmse:.4f} yards")
    print(f"   Total time: {total_training_time/60:.1f} minutes")

    return best_val_rmse, training_history

# ============================================================================
# MAIN TRAINING RUNNER
# ============================================================================

def run_training():
    """Run the complete training pipeline"""
    print("\n" + "="*80)
    print("🚀 NFL TRAINING PIPELINE - v10")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"✅ LOOKBACK: {LOOKBACK} frames")
    print(f"✅ Loss Function: CleanPositionLoss")
    print(f"🆕 Features: Advanced v2 Features")
    print(f"✅ Data Augmentation: Enabled (noise level: {NOISE_LEVEL})")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessor
    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.pt"
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run preprocessing first!")

    preprocessor_params = torch.load(preprocessor_path)
    input_dim = len(preprocessor_params['feature_cols'])

    print(f"[INFO] Preprocessor loaded: {input_dim} features")
    print(f"[INFO] 📝 Logs will be saved to: {LOGS_DIR}")

    # Load datasets
    train_dataset = FastNFLTrajectoryDataset(PROCESSED_DATA_DIR / "train")
    val_dataset = FastNFLTrajectoryDataset(PROCESSED_DATA_DIR / "val")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, 
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, 
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=PERSISTENT_WORKERS
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}")

    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {len(ENSEMBLE_SEEDS)} MODELS")
    print(f"{'='*80}")

    ensemble_results = []
    total_start = time.time()

    for idx, seed in enumerate(ENSEMBLE_SEEDS, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {idx}/{len(ENSEMBLE_SEEDS)} - SEED {seed}")
        print(f"{'#'*80}")

        model_start = time.time()
        best_rmse, history = train_single_model(
            seed, train_loader, val_loader, preprocessor_params, input_dim
        )
        model_time = time.time() - model_start

        ensemble_results.append({
            'seed': seed,
            'best_val_rmse': best_rmse,
            'model_path': str(MODELS_DIR / f"model_seed{seed}.pth"),
            'training_time_minutes': model_time / 60
        })

        print(f"\n⏱️ Model {idx} time: {model_time/60:.1f} minutes")

    total_time = time.time() - total_start

    create_combined_log(LOGS_DIR, ensemble_results)

    print(f"\n\n{'='*80}")
    print(f"🏆 ENSEMBLE TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n⏱️ Total Time: {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")

    print(f"\n📊 Individual Model Performance:")
    print(f"{'─'*80}")
    for result in ensemble_results:
        print(f"   Seed {result['seed']:>3}: RMSE = {result['best_val_rmse']:>6.4f} yd | Time: {result['training_time_minutes']:>5.1f}min")

    rmse_values = [r['best_val_rmse'] for r in ensemble_results]
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    min_rmse = np.min(rmse_values)
    max_rmse = np.max(rmse_values)

    print(f"\n📈 Ensemble Statistics:")
    print(f"{'─'*80}")
    print(f"   Mean RMSE:     {mean_rmse:>6.4f} yards")
    print(f"   Std Dev:       {std_rmse:>6.4f} yards")
    print(f"   Best Model:    {min_rmse:>6.4f} yards")
    print(f"   Worst Model:   {max_rmse:>6.4f} yards")

    expected_ensemble_rmse = mean_rmse * 0.95
    print(f"\n🎯 Expected Ensemble Performance:")
    print(f"{'─'*80}")
    print(f"   Estimated RMSE: {expected_ensemble_rmse:>6.4f} yards")

    ensemble_metadata = {
        'num_models': len(ENSEMBLE_SEEDS),
        'seeds': ENSEMBLE_SEEDS,
        'results': ensemble_results,
        'statistics': {
            'mean_rmse': float(mean_rmse),
            'std_rmse': float(std_rmse),
            'min_rmse': float(min_rmse),
            'max_rmse': float(max_rmse),
            'expected_ensemble_rmse': float(expected_ensemble_rmse)
        },
        'total_training_time_hours': total_time / 3600,
        'logs_directory': str(LOGS_DIR),
        'config': {
            'lookback': LOOKBACK,
            'd_model': D_MODEL,
            'n_layers': N_LAYERS,
            'curriculum_learning': False,
            'zone_coverage_features': True
        }
    }

    metadata_path = MODELS_DIR / "ensemble_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)

    print(f"\n💾 Outputs:")
    print(f"   Models: {MODELS_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"   Metadata: {metadata_path}")
    print(f"\n✨ Ready for inference!")

    return ensemble_results, ensemble_metadata

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("""
    ================================================================================
    🏈 NFL TRAINING PIPELINE - v10 (Feature Test)
    ================================================================================
    Testing advanced v2 features on the stable v9 pipeline.
    
    - ✅ Base Pipeline: v9
    - ✅ LOOKBACK = 25
    - ✅ Loss Function: CleanPositionLoss (MSE)
    - 🆕 FEATURES: Zone Coverage & Advanced Ball Trajectory
    - 🧪 SEEDS: [42,43,44,45,46,47,48,49,50,51,52]
    ================================================================================
    """)

    print(f"Device: {DEVICE}")
    print(f"📊 Max Epochs: {EPOCHS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🎓 Learning Rate: {LEARNING_RATE}")
    
    try:
        results, metadata = run_training()

        print(f"\n\n{'='*80}")
        print(f"🎉 SUCCESS! Training complete!")
        print(f"{'='*80}")
        print(f"✅ Trained {len(results)} models: {MODELS_DIR}")
        print(f"✅ Training logs: {LOGS_DIR}")
        print(f"\n🚀 Next: Run inference or compare to v9 baseline!")
        print(f"{'='*80}\n")

        return results, metadata

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()