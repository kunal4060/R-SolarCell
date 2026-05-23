import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import copy

import warnings
warnings.filterwarnings('ignore')

FONT_SIZES = {
    "title": 16,
    "label": 16,
    "tick": 11,
    "legend": 11,
    "annotation": 10,
    "title_small": 12,
}
THRESHOLD = 0.15


def load_and_preprocess(csv_path: str):
    """Load CSV and preprocess data."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Output parameters
    output_cols = ['Voc', 'Jsc', 'FF', 'eta']
    
    # Check if output columns exist
    missing_cols = [col for col in output_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing output columns: {missing_cols}")
    
    # Input features: all columns except output columns and metadata
    exclude_cols = output_cols + ['source_file']
    input_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Input features ({len(input_cols)}): {input_cols[:5]}... (showing first 5)")
    print(f"Output features: {output_cols}")
    
    # Remove rows with NaN values
    df = df.dropna(subset=output_cols + input_cols)
    print(f"Dataset after removing NaN: {df.shape}")
    
    X = df[input_cols].values
    y = df[output_cols].values
    
    return X, y, input_cols, output_cols


class ANNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class CNNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)


class RNNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, h = self.rnn(x)
        return self.fc(h[-1])


def build_neural_network(model_name: str, input_dim: int, output_dim: int = 4):
    """Build a neural network for multi-output regression."""
    name = model_name.strip().upper()
    if name == "ANN":
        return ANNModel(input_dim, output_dim)
    if name == "CNN":
        return CNNModel(input_dim, output_dim)
    if name == "RNN":
        return RNNModel(input_dim, output_dim)
    raise ValueError(f"Unsupported model type: {model_name}")


def display_target_name(name: str) -> str:
    """Human-friendly label for plots."""
    return "Efficiency" if name == "eta" else name


def accuracy_within_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = THRESHOLD) -> float:
    """Percent of predictions within +/- threshold of actual values."""
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    within = np.abs(y_pred - y_true) <= (threshold * denom)
    return float(np.mean(within))


def train_model(model_name, X_train, X_test, y_train, y_test, output_cols):
    """Train the neural network model."""
    
    # Normalize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler_X.transform(X_test).astype(np.float32)
    
    # Normalize outputs
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).astype(np.float32)
    y_test_scaled = scaler_y.transform(y_test).astype(np.float32)
    
    # Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_scaled))
    val_dataset = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test_scaled))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Build model
    model = build_neural_network(model_name=model_name, input_dim=X_train_scaled.shape[1])
    print(f"\nModel Summary ({model_name}):\n{model}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Training Loop
    epochs = 200
    patience = 20
    best_val_loss = float('inf')
    early_stop_count = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    print(f"\nTraining Neural Network ({model_name})...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_mae = 0.0, 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_mae += torch.nn.functional.l1_loss(y_pred, y_batch).item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.nn.functional.l1_loss(y_pred, y_batch).item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    model.load_state_dict(best_model_weights)
    
    # Evaluate
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_test_scaled)).numpy()
        
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    overall_accuracy = accuracy_within_threshold(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("TEST SET METRICS")
    print(f"{'='*60}")
    print(f"Overall MSE:  {mse:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"Overall MAE:  {mae:.6f}")
    print(f"Overall R²:   {r2:.6f}")
    print(f"Overall Acc@10%: {overall_accuracy:.4f}")
    print(f"{'='*60}\n")
    
    # Per-output metrics
    metrics = {}
    for i, col in enumerate(output_cols):
        mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
        rmse_i = np.sqrt(mse_i)
        mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2_i = r2_score(y_test[:, i], y_pred[:, i])
        acc_i = accuracy_within_threshold(y_test[:, i], y_pred[:, i])
        
        metrics[col] = {
            'mse': float(mse_i),
            'rmse': float(rmse_i),
            'mae': float(mae_i),
            'r2': float(r2_i),
            'accuracy': float(acc_i)
        }
        
        print(f"{col:5} | RMSE: {rmse_i:.6f} | MAE: {mae_i:.6f} | R²: {r2_i:.6f} | Acc@10%: {acc_i:.4f}")

    metrics["overall"] = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "accuracy": float(overall_accuracy),
    }
    
    # Save model and scalers
    model_dir = Path("neural_network_results") / model_name.lower()
    model_path = model_dir / "neural_network_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save scalers
    import pickle
    with open(model_path.parent / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(model_path.parent / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    
    # Save metrics
    with open(model_path.parent / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {model_path.parent / 'model_metrics.json'}")
    
    # Save predictions
    predictions_df = pd.DataFrame(y_pred, columns=[f"{col}_pred" for col in output_cols])
    for col in output_cols:
        predictions_df[f"{col}_actual"] = y_test[:, output_cols.index(col)]
    predictions_df.to_csv(model_path.parent / "test_predictions.csv", index=False)
    print(f"Predictions saved: {model_path.parent / 'test_predictions.csv'}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=FONT_SIZES["label"])
    axes[0].set_ylabel('Loss (MSE)', fontsize=FONT_SIZES["label"])
    axes[0].set_title('Training and Validation Loss', fontsize=FONT_SIZES["title"])
    axes[0].legend(fontsize=FONT_SIZES["legend"])
    axes[0].tick_params(labelsize=FONT_SIZES["tick"])
    axes[0].grid(True)
    
    axes[1].plot(history['mae'], label='Train MAE')
    axes[1].plot(history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch', fontsize=FONT_SIZES["label"])
    axes[1].set_ylabel('MAE', fontsize=FONT_SIZES["label"])
    axes[1].set_title('Training and Validation MAE', fontsize=FONT_SIZES["title"])
    axes[1].legend(fontsize=FONT_SIZES["legend"])
    axes[1].tick_params(labelsize=FONT_SIZES["tick"])
    axes[1].grid(True)
    
    plt.tight_layout()
    fig.suptitle(f"{model_name} Training History", fontsize=FONT_SIZES["title"])
    plt.savefig(model_path.parent / "training_history.png", dpi=100)
    plt.close()
    
    # ---------------------------------------------------------
    # NEW GRAPHS: Actual vs Predicted Scatter plots
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(output_cols):
        ax = axes[i]
        actual = y_test[:, i]
        predicted = y_pred[:, i]
        ax.scatter(actual, predicted, alpha=0.5, color='royalblue')
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        
        col_label = display_target_name(col)
        ax.set_title(
            (
                f"{model_name} - {col_label} Actual Vs Predicted\n"
                f"R² = {metrics[col]['r2']:.4f} | RMSE = {metrics[col]['rmse']:.4f} | "
                f"MAE = {metrics[col]['mae']:.4f} | Accuracy = {metrics[col]['accuracy']:.4f}"
            ),
            fontsize=FONT_SIZES["title_small"],
        )
        ax.set_xlabel(f'Actual {col_label}', fontsize=FONT_SIZES["label"])
        ax.set_ylabel(f'Predicted {col_label}', fontsize=FONT_SIZES["label"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig(model_path.parent / "actual_vs_predicted.png", dpi=100)
    plt.close()
    
    # ---------------------------------------------------------
    # NEW GRAPHS: Bar charts for R², RMSE, MAE, Accuracy
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    def plot_metric_bar(ax, values, title, ylabel, color, ylim=None):
        display_cols = [display_target_name(col) for col in output_cols]
        ax.bar(display_cols, values, color=color)
        ax.set_title(title, fontsize=FONT_SIZES["title"])
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        offset = 0.02 if ylim is not None else max(values) * 0.02
        for i, v in enumerate(values):
            ax.text(
                i,
                v + offset,
                f'{v:.3f}',
                ha='center',
                fontweight='bold',
                fontsize=FONT_SIZES["annotation"],
            )

    r2_vals = [metrics[col]['r2'] for col in output_cols]
    rmse_vals = [metrics[col]['rmse'] for col in output_cols]
    mae_vals = [metrics[col]['mae'] for col in output_cols]
    acc_vals = [metrics[col]['accuracy'] for col in output_cols]

    plot_metric_bar(
        axes[0],
        r2_vals,
        f'{model_name} - R² Score by Output Parameter',
        'R² Score',
        'mediumseagreen',
        ylim=(0, 1.05),
    )
    plot_metric_bar(
        axes[1],
        rmse_vals,
        f'{model_name} - RMSE by Output Parameter',
        'RMSE',
        'coral',
    )
    plot_metric_bar(
        axes[2],
        mae_vals,
        f'{model_name} - MAE by Output Parameter',
        'MAE',
        'slateblue',
    )
    plot_metric_bar(
        axes[3],
        acc_vals,
        f'{model_name} - Accuracy by Output Parameter',
        'Accuracy',
        'teal',
        ylim=(0, 1.05),
    )

    plt.tight_layout()
    plt.savefig(model_path.parent / "metrics_bar_charts.png", dpi=100)
    plt.close()
    
    print(f"Training plot saved: {model_path.parent / 'training_history.png'}")
    print(f"Scatter plots saved: {model_path.parent / 'actual_vs_predicted.png'}")
    print(f"Metrics charts saved: {model_path.parent / 'metrics_bar_charts.png'}")
    
    return model, scaler_X, scaler_y, y_pred


def plot_nn_comparison(metrics_rows: list[dict], out_dir: Path) -> None:
    """Plot comparison charts across ANN/CNN/RNN by target and metric."""
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["target_label"] = metrics_df["target"].apply(display_target_name)
    targets_only = metrics_df[metrics_df["target"] != "overall"].copy()

    for metric in ["R2", "RMSE", "MAE", "Accuracy"]:
        pivot = targets_only.pivot(index="model", columns="target_label", values=metric)
        ax = pivot.plot(kind="bar", figsize=(11, 6))
        metric_label = "R²" if metric == "R2" else metric
        ax.set_title(f"NN Comparison - {metric_label}", fontsize=FONT_SIZES["title"])
        ax.set_xlabel("Model", fontsize=FONT_SIZES["label"])
        ax.set_ylabel(metric_label, fontsize=FONT_SIZES["label"])
        ax.legend(
            title="Target",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=FONT_SIZES["legend"],
            title_fontsize=FONT_SIZES["legend"],
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        plt.tight_layout()
        plt.savefig(out_dir / f"nn_model_performance_{metric.lower()}.png", dpi=200)
        plt.close()

    overall = metrics_df[metrics_df["target"] == "overall"].set_index("model")
    overall_plot = overall.rename(columns={"R2": "R²"})
    ax = overall_plot[["R²", "RMSE", "MAE", "Accuracy"]].plot(kind="bar", figsize=(10, 5))
    ax.set_title("NN Overall Performance (Average Across Targets)", fontsize=FONT_SIZES["title"])
    ax.set_xlabel("Model", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Score", fontsize=FONT_SIZES["label"])
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=FONT_SIZES["legend"])
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    plt.tight_layout()
    plt.savefig(out_dir / "nn_model_performance_overall.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent / "part1_2_3_4_6_7_8_9_10_combined.csv"
    
    # Load and preprocess
    X, y, input_cols, output_cols = load_and_preprocess(csv_path)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    predictions_out = pd.DataFrame(y_test, columns=[f"actual_{col}" for col in output_cols])
    for model_name in ["ANN", "CNN", "RNN"]:
        model, scaler_X, scaler_y, y_pred = train_model(
            model_name, X_train, X_test, y_train, y_test, output_cols
        )
        pred_df = pd.DataFrame(y_pred, columns=[f"pred_{model_name}_{col}" for col in output_cols])
        predictions_out = pd.concat([predictions_out, pred_df], axis=1)

    results_root = Path(__file__).resolve().parents[1]
    nn_outputs_path = results_root / "neural_network_results" / "nn_test_predictions.csv"
    nn_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.to_csv(nn_outputs_path, index=False)
    print(f"\nNeural network combined predictions saved: {nn_outputs_path}")

    combined_performance_path = results_root / "combined_model_performance.csv"
    performance_rows = []

    ml_metrics_path = results_root / "ml_training" / "results_part1_2_3_4_6_7_8_9_10" / "model_metrics.csv"
    if ml_metrics_path.exists():
        ml_metrics = pd.read_csv(ml_metrics_path)
        for _, row in ml_metrics.iterrows():
            performance_rows.append(
                {
                    "source": "ML",
                    "model": row["model"],
                    "target": row["target"],
                    "R2": row["R2"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "Accuracy": row.get("Accuracy", np.nan),
                }
            )
    else:
        print("Combined performance not created: ML metrics file not found.")

    nn_metrics_rows = []
    for model_name in ["ann", "cnn", "rnn"]:
        metrics_path = results_root / "neural_network_results" / model_name / "model_metrics.json"
        if not metrics_path.exists():
            print(f"Combined performance skipped: missing {metrics_path}")
            continue
        with open(metrics_path, "r") as f:
            nn_metrics = json.load(f)

        for target_name, values in nn_metrics.items():
            nn_metrics_rows.append(
                {
                    "source": "NN",
                    "model": model_name.upper(),
                    "target": target_name,
                    "R2": values.get("r2"),
                    "RMSE": values.get("rmse"),
                    "MAE": values.get("mae"),
                    "Accuracy": values.get("accuracy"),
                }
            )

    if nn_metrics_rows:
        nn_metrics_df = pd.DataFrame(nn_metrics_rows)
        nn_metrics_df.to_csv(results_root / "neural_network_results" / "nn_model_metrics.csv", index=False)
        plot_nn_comparison(nn_metrics_rows, results_root / "neural_network_results")
        performance_rows.extend(nn_metrics_rows)

    if performance_rows:
        combined_df = pd.DataFrame(performance_rows)
        combined_df.to_csv(combined_performance_path, index=False)
        print(f"Combined model performance saved: {combined_performance_path}")
    
    print("\n✓ Training complete!")
