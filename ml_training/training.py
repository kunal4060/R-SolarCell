from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
	from xgboost import XGBRegressor
except ImportError as exc:
	raise ImportError(
		"xgboost is not installed. Install it with: pip install xgboost"
	) from exc


# Input columns requested by user
FEATURE_COLUMNS = [
	"PEDOT-HTL (L1)>>thickness",
	"PEDOT-HTL (L1)>>band gap",
	"PEDOT-HTL (L1)>>electron affinity",
	"N-CS2AgBiBr6 (L2)>>thickness",
	"N-CS2AgBiBr6 (L2)>>band gap",
	"N-CS2AgBiBr6 (L2)>>electron affinity",
	"P-CsPb(I0.6Br0.4)3 (L3)>>thickness",
	"P-CsPb(I0.6Br0.4)3 (L3)>>band gap",
	"P-CsPb(I0.6Br0.4)3 (L3)>>electron affinity",
]

# Targets requested by user
TARGET_COLUMNS = ["Voc", "Jsc", "FF", "eta"]


def build_models() -> dict[str, object]:
	"""Create model dictionary for benchmarking."""
	return {
		"XGBoost": MultiOutputRegressor(
			XGBRegressor(
				n_estimators=300,
				learning_rate=0.05,
				max_depth=6,
				subsample=0.9,
				colsample_bytree=0.9,
				reg_lambda=1.0,
				random_state=42,
				objective="reg:squarederror",
				n_jobs=-1,
			)
		),
		"RandomForest": RandomForestRegressor(
			n_estimators=500,
			random_state=42,
			n_jobs=-1,
		),
		"GradientBoosting": MultiOutputRegressor(
			GradientBoostingRegressor(random_state=42)
		),
		"LinearRegression": Pipeline(
			[
				("scaler", StandardScaler()),
				("model", LinearRegression()),
			]
		),
		"RidgeRegression": Pipeline(
			[
				("scaler", StandardScaler()),
				("model", Ridge(alpha=1.0)),
			]
		),
		"KNeighborsRegressor": Pipeline(
			[
				("scaler", StandardScaler()),
				("model", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
			]
		),
	}


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, float]]]:
	"""Return overall and per-target metrics."""
	per_target_metrics: list[dict[str, float]] = []
	for i, target in enumerate(TARGET_COLUMNS):
		target_true = y_true[:, i]
		target_pred = y_pred[:, i]
		per_target_metrics.append(
			{
				"target": target,
				"R2": r2_score(target_true, target_pred),
				"RMSE": float(np.sqrt(mean_squared_error(target_true, target_pred))),
				"MAE": mean_absolute_error(target_true, target_pred),
			}
		)

	overall = {
		"target": "overall",
		"R2": np.mean([m["R2"] for m in per_target_metrics]),
		"RMSE": np.mean([m["RMSE"] for m in per_target_metrics]),
		"MAE": np.mean([m["MAE"] for m in per_target_metrics]),
	}
	return overall, per_target_metrics


def plot_model_performance(metrics_df: pd.DataFrame, out_dir: Path) -> None:
	"""Create and save bar plots comparing models by target and metric."""
	targets_only = metrics_df[metrics_df["target"] != "overall"].copy()

	for metric in ["R2", "RMSE", "MAE"]:
		pivot = targets_only.pivot(index="model", columns="target", values=metric)
		ax = pivot.plot(kind="bar", figsize=(11, 6))
		ax.set_title(f"Model Comparison - {metric}")
		ax.set_xlabel("Model")
		ax.set_ylabel(metric)
		ax.legend(title="Target", bbox_to_anchor=(1.02, 1), loc="upper left")
		plt.tight_layout()
		plt.savefig(out_dir / f"model_performance_{metric.lower()}.png", dpi=200)
		plt.close()

	overall = metrics_df[metrics_df["target"] == "overall"].set_index("model")
	ax = overall[["R2", "RMSE", "MAE"]].plot(kind="bar", figsize=(10, 5))
	ax.set_title("Overall Model Performance (Average Across Targets)")
	ax.set_xlabel("Model")
	ax.set_ylabel("Score")
	ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
	plt.tight_layout()
	plt.savefig(out_dir / "model_performance_overall.png", dpi=200)
	plt.close()


def plot_predictions_vs_actual(
	model_name: str,
	y_test: np.ndarray,
	y_pred: np.ndarray,
	target_columns: list[str],
	out_dir: Path
) -> None:
	"""Create scatter plots of predicted vs actual for each target variable."""
	for i, target in enumerate(target_columns):
		actual = y_test[:, i]
		predicted = y_pred[:, i]
		
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.scatter(actual, predicted, alpha=0.6, edgecolors='k', linewidth=0.5)
		
		# Add perfect prediction line
		min_val = min(actual.min(), predicted.min())
		max_val = max(actual.max(), predicted.max())
		ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
		
		# Calculate R²
		r2 = r2_score(actual, predicted)
		rmse = np.sqrt(mean_squared_error(actual, predicted))
		
		ax.set_xlabel(f"Actual {target}", fontsize=12)
		ax.set_ylabel(f"Predicted {target}", fontsize=12)
		ax.set_title(f"{model_name} - {target} (R² = {r2:.4f}, RMSE = {rmse:.4f})", fontsize=13)
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(out_dir / f"predicted_vs_actual_{model_name}_{target}.png", dpi=200)
		plt.close()


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	data_path = script_dir / "part1_2_3_4_6_7_8_9_10_combined.csv"
	output_dir = script_dir / "results_part1_2_3_4_6_7_8_9_10"
	output_dir.mkdir(exist_ok=True)

	if not data_path.exists():
		raise FileNotFoundError(f"Input CSV not found: {data_path}")

	df = pd.read_csv(data_path)

	missing_columns = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col not in df.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns in input CSV: {missing_columns}")

	clean_df = df[FEATURE_COLUMNS + TARGET_COLUMNS].dropna().copy()

	X = clean_df[FEATURE_COLUMNS]
	y = clean_df[TARGET_COLUMNS]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
	)

	models = build_models()

	all_rows: list[dict[str, float | str]] = []
	predictions_out = y_test.reset_index(drop=True).copy()
	predictions_out.columns = [f"actual_{c}" for c in predictions_out.columns]

	for model_name, model in models.items():
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		overall, per_target = evaluate_predictions(y_test.values, y_pred)
		overall["model"] = model_name
		all_rows.append(overall)

		for row in per_target:
			row["model"] = model_name
			all_rows.append(row)

		pred_df = pd.DataFrame(y_pred, columns=[f"pred_{model_name}_{c}" for c in TARGET_COLUMNS])
		predictions_out = pd.concat([predictions_out, pred_df], axis=1)
		
		# Generate predicted vs actual plots for this model
		plot_predictions_vs_actual(model_name, y_test.values, y_pred, TARGET_COLUMNS, output_dir)

	metrics_df = pd.DataFrame(all_rows)[["model", "target", "R2", "RMSE", "MAE"]]
	metrics_df = metrics_df.sort_values(["target", "R2"], ascending=[True, False]).reset_index(drop=True)

	metrics_csv = output_dir / "model_metrics.csv"
	preds_csv = output_dir / "test_set_predictions.csv"
	metrics_df.to_csv(metrics_csv, index=False)
	predictions_out.to_csv(preds_csv, index=False)

	plot_model_performance(metrics_df, output_dir)

	print(f"Samples used after cleaning: {len(clean_df)}")
	print(f"Train/Test split: {len(X_train)} / {len(X_test)} (80/20)")
	print(f"Saved metrics to: {metrics_csv}")
	print(f"Saved test predictions to: {preds_csv}")
	print("\nComparison plots (by model and metric):")
	print(f" - {output_dir / 'model_performance_r2.png'}")
	print(f" - {output_dir / 'model_performance_rmse.png'}")
	print(f" - {output_dir / 'model_performance_mae.png'}")
	print(f" - {output_dir / 'model_performance_overall.png'}")
	print("\nPredicted vs Actual plots (by model and target):")
	for model_name in models.keys():
		for target in TARGET_COLUMNS:
			print(f" - {output_dir / f'predicted_vs_actual_{model_name}_{target}.png'}")


if __name__ == "__main__":
	main()
