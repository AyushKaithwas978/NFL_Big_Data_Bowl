# 🏈 NFL Big Data Bowl - GNN-Transformer Trajectory Forecasting

A full-stack spatiotemporal modeling pipeline for the NFL Big Data Bowl (2026). This project combines physics-aware feature engineering, graph neural networks, and transformer encoders to predict defensive player trajectories and ball landing outcomes at frame-level resolution.

## ✨ Highlights
- Built an end-to-end pipeline: preprocessing -> training -> inference with reproducible artifacts and logs.
- Engineered advanced v2 features: physics vectors, interaction dynamics, zone coverage, temporal derivatives, and ball-flight modeling.
- Designed a hybrid model that fuses Transformer sequence encoders, GNN-based interaction modeling, and GRU decoding.
- Implemented inference-time ensembling with adaptive weighting, test-time augmentation, and post-processing constraints.

## 🧠 Architecture Snapshot
- Preprocessing: Generates dense feature sets per player and packages plays into graph-aware tensors.
- Model: Transformer encodes each player's recent motion history; a GNN aggregates team dynamics; a GRU decoder forecasts positions autoregressively.
- Ball Trajectory Head: Predicts ball landing coordinates and confidence to guide downstream decoding.
- Inference: Weighted ensemble averaging with optional TTA (flip + rotation) and trajectory smoothing.

## 🗂️ Repository Structure
- `pipeline/v10_preprocessing.py`
- `pipeline/v10_training_cosinelr.py`
- `pipeline/v10_inference_cosinelr.py`

## 📊 Data
This repo does not include any competition data files.

Download the dataset from Kaggle and place the training CSVs according to your local layout. The preprocessing script expects weekly files in the following pattern by default:
- `../train/input_2023_w*.csv`
- `../train/output_2023_w*.csv`

If your paths differ, update the glob patterns near the top of `pipeline/v10_preprocessing.py`.

## ⚙️ Quickstart (Local)
1. Update the data paths in `pipeline/v10_preprocessing.py`.
2. Run preprocessing to generate train/val `.pt` files and the preprocessor:

```bash
python pipeline/v10_preprocessing.py
```

3. Update output directories in `pipeline/v10_training_cosinelr.py` and train:

```bash
python pipeline/v10_training_cosinelr.py
```

4. Use `pipeline/v10_inference_cosinelr.py` for Kaggle inference (paths are configured for Kaggle datasets by default). Update `MODELS_DIR` and `PREPROCESSOR_PATH` if you run locally.

## 🔬 Key Technical Details
- Lookback window: 25 frames
- Feature stack: physics vectors, pursuit/coverage metrics, route embeddings, zone coverage, temporal derivatives
- Graph edges: proximity-based (distance < 30 yards) with exponential weighting
- Loss: player trajectory MSE + weighted ball landing loss
- Metrics: RMSE, ADE, FDE (in yards)

## 📌 Resume-Ready Highlights
- Built a production-grade spatiotemporal modeling pipeline with feature-rich preprocessing and automated artifact generation.
- Integrated Transformer + GNN + GRU architectures to capture multi-agent dynamics in high-frequency tracking data.
- Engineered inference-time robustness with adaptive ensemble weighting, augmentation, and physical constraints.

## 📝 Notes
- Training and inference scripts include hard-coded paths (Kaggle/Colab style). Update them to match your environment before running.
- Metrics and logs are emitted to CSV/JSON files during training for reproducibility.

## 📜 License
This project is licensed under the terms in `LICENSE`.
