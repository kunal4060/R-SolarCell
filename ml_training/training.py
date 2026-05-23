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

FONT_SIZES = {
	"title": 16,
	"label": 16,
	"tick": 11,
	"legend": 11,
	"annotation": 10,
	"title_small": 12,
}
THRESHOLD = 0.15


def accuracy_within_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = THRESHOLD) -> float:
	"""Percent of predictions within +/- threshold of actual values."""
	eps = 1e-8
	denom = np.maximum(np.abs(y_true), eps)
	within = np.abs(y_pred - y_true) <= (threshold * denom)
	return float(np.mean(within))


def display_target_name(name: str) -> str:
	"""Human-friendly label for plots."""
	return "Efficiency" if name == "eta" else name


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
				"Accuracy": accuracy_within_threshold(target_true, target_pred, threshold=0.10),
			}
		)

	overall = {
		"target": "overall",
		"R2": np.mean([m["R2"] for m in per_target_metrics]),
		"RMSE": np.mean([m["RMSE"] for m in per_target_metrics]),
		"MAE": np.mean([m["MAE"] for m in per_target_metrics]),
		"Accuracy": np.mean([m["Accuracy"] for m in per_target_metrics]),
	}
	return overall, per_target_metrics


def plot_model_performance(metrics_df: pd.DataFrame, out_dir: Path) -> None:
	"""Create and save bar plots comparing models by target and metric."""
	targets_only = metrics_df[metrics_df["target"] != "overall"].copy()
	targets_only["target_label"] = targets_only["target"].apply(display_target_name)

	for metric in ["R2", "RMSE", "MAE", "Accuracy"]:
		pivot = targets_only.pivot(index="model", columns="target_label", values=metric)
		ax = pivot.plot(kind="bar", figsize=(11, 6))
		metric_label = "R\u00b2" if metric == "R2" else metric
		ax.set_title(f"Model Comparison - {metric_label}", fontsize=FONT_SIZES["title"])
		ax.set_xlabel("Model", fontsize=FONT_SIZES["label"])
		ax.set_ylabel(metric_label, fontsize=FONT_SIZES["label"])
		ax.legend(title="Target", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=FONT_SIZES["legend"], title_fontsize=FONT_SIZES["legend"])
		ax.tick_params(labelsize=FONT_SIZES["tick"])
		plt.tight_layout()
		plt.savefig(out_dir / f"model_performance_{metric.lower()}.png", dpi=200)
		plt.close()

	overall = metrics_df[metrics_df["target"] == "overall"].set_index("model")
	overall_plot = overall.rename(columns={"R2": "R\u00b2"})
	ax = overall_plot[["R\u00b2", "RMSE", "MAE", "Accuracy"]].plot(kind="bar", figsize=(10, 5))
	ax.set_title("Overall Model Performance (Average Across Targets)", fontsize=FONT_SIZES["title"])
	ax.set_xlabel("Model", fontsize=FONT_SIZES["label"])
	ax.set_ylabel("Score", fontsize=FONT_SIZES["label"])
	ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=FONT_SIZES["legend"])
	ax.tick_params(labelsize=FONT_SIZES["tick"])
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
	"""Create a 2x2 scatter plot of predicted vs actual for each target."""
	fig, axes = plt.subplots(2, 2, figsize=(12, 10))
	axes = axes.flatten()
	for i, target in enumerate(target_columns):
		ax = axes[i]
		actual = y_test[:, i]
		predicted = y_pred[:, i]
		ax.scatter(actual, predicted, alpha=0.6, edgecolors='k', linewidth=0.5)

		# Add perfect prediction line
		min_val = min(actual.min(), predicted.min())
		max_val = max(actual.max(), predicted.max())
		ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

		# Metrics for title
		r2 = r2_score(actual, predicted)
		rmse = np.sqrt(mean_squared_error(actual, predicted))
		mae = mean_absolute_error(actual, predicted)
		acc = accuracy_within_threshold(actual, predicted, threshold=0.10)
		target_label = display_target_name(target)

		ax.set_xlabel(f"Actual {target_label}", fontsize=FONT_SIZES["label"])
		ax.set_ylabel(f"Predicted {target_label}", fontsize=FONT_SIZES["label"])
		ax.set_title(
			(
				f"{model_name} - {target_label} Actual Vs Predicted\n"
				f"R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f} | "
				f"Accuracy = {acc:.4f}"
			),
			fontsize=FONT_SIZES["title_small"],
		)
		ax.legend(fontsize=FONT_SIZES["legend"])
		ax.tick_params(labelsize=FONT_SIZES["tick"])
		ax.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(out_dir / f"actual_vs_predicted_{model_name}.png", dpi=200)
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

	metrics_df = pd.DataFrame(all_rows)[["model", "target", "R2", "RMSE", "MAE", "Accuracy"]]
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
	print("\nActual vs Predicted plots (by model):")
	for model_name in models.keys():
		print(f" - {output_dir / f'actual_vs_predicted_{model_name}.png'}")


if __name__ == "__main__":
	main()
