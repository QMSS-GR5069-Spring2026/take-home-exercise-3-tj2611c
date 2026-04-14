import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = pd.read_csv("results (1).csv")
races = pd.read_csv("races.csv")
qualifying = pd.read_csv("qualifying.csv")

def time_to_seconds(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x == r"\N":
        return np.nan
    try:
        if ":" in x:
            mins, secs = x.split(":")
            return float(mins) * 60 + float(secs)
        return float(x)
    except:
        return np.nan

for col in ["q1", "q2", "q3"]:
    qualifying[col + "_sec"] = qualifying[col].apply(time_to_seconds)

qualifying_small = qualifying[
    ["raceId", "driverId", "constructorId", "position", "q1_sec", "q2_sec", "q3_sec"]
].rename(columns={"position": "quali_position"})

df = results.merge(
    races[["raceId", "year", "round", "circuitId"]],
    on="raceId",
    how="left"
)

df = df.merge(
    qualifying_small,
    on=["raceId", "driverId", "constructorId"],
    how="left"
)

numeric_cols = [
    "grid", "positionOrder", "laps", "milliseconds", "fastestLap",
    "rank", "fastestLapSpeed", "year", "round", "circuitId",
    "driverId", "constructorId", "quali_position", "q1_sec", "q2_sec", "q3_sec", "points"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

target = "points"

features = [
    "grid",
    "laps",
    "milliseconds",
    "fastestLap",
    "rank",
    "fastestLapSpeed",
    "year",
    "round",
    "circuitId",
    "driverId",
    "constructorId",
    "quali_position",
    "q1_sec",
    "q2_sec",
    "q3_sec"
]

model_df = df[features + [target]].copy()

for col in model_df.columns:
    if model_df[col].dtype.kind in "biufc":
        model_df[col] = model_df[col].fillna(model_df[col].median())

model_df = model_df.dropna(subset=[target])

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("F1_Homework4_RandomForest_Points")

param_grid = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

all_results = []
run_count = 0

for params in ParameterGrid(param_grid):
    if run_count >= 10:
        break

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("target", target)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "random_forest_model")

        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, linestyle="--")
        plt.xlabel("Predicted Points")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        residual_plot_path = f"residuals_run_{run_count+1}.png"
        plt.tight_layout()
        plt.savefig(residual_plot_path)
        plt.close()
        mlflow.log_artifact(residual_plot_path)

        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        fi_path = f"feature_importance_run_{run_count+1}.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        pred_df = pd.DataFrame({
            "actual_points": y_test.values,
            "predicted_points": y_pred,
            "residual": residuals.values
        })
        pred_path = f"predictions_run_{run_count+1}.csv"
        pred_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path)

        run_info = {
            "run_number": run_count + 1,
            "run_id": mlflow.active_run().info.run_id,
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf": params["min_samples_leaf"],
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
        all_results.append(run_info)

        run_count += 1

results_df = pd.DataFrame(all_results).sort_values(
    by=["rmse", "mae", "r2"],
    ascending=[True, True, False]
)

print("Top experiment results:")
print(results_df.head(10))

best_run = results_df.iloc[0]
print("\nBest run summary:")
print(best_run)

results_df.to_csv("all_mlflow_runs_summary.csv", index=False)

best_model_explanation = f"""
Best Model Selection

I selected run {int(best_run['run_number'])} as the best model because it achieved the lowest RMSE and one of the strongest overall error profiles across all experiments.
The best run used n_estimators={best_run['n_estimators']}, max_depth={best_run['max_depth']}, min_samples_split={best_run['min_samples_split']}, and min_samples_leaf={best_run['min_samples_leaf']}.
Its performance was MAE={best_run['mae']:.4f}, MSE={best_run['mse']:.4f}, RMSE={best_run['rmse']:.4f}, and R2={best_run['r2']:.4f}.
I selected this run because lower MAE, MSE, and RMSE indicate better prediction accuracy, while a higher R2 shows that the model explains more variation in race points.
Compared with the other runs, this model provided the best balance between accuracy and generalization on the test set.
"""

with open("best_model_explanation.txt", "w", encoding="utf-8") as f:
    f.write(best_model_explanation)

print(best_model_explanation)

print("""
Take these screenshots for submission:
1. MLflow homepage
2. Experiment page showing at least 10 runs
3. Best run page showing parameters and metrics
4. Artifacts page showing residual plot and feature importance csv
""")