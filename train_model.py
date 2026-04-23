from os import name
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import mlflow
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeRegressor
import joblib


def scale_frame(frame):
    df = frame.copy()
    X,y = df.drop(columns = ['math score']), df['math score']
    print(f"Features being used: {X.columns.tolist()}")
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1,1))
    return X_scale, Y_scale, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    df = pd.read_csv("./df_clear.csv")
    X,Y, power_trans = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42)
    

    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1 ],
      'l1_ratio': [0.001, 0.05, 0.01, 0.2]
    }
    mlflow.set_experiment("linear model students' math score")
    with mlflow.start_run(run_name="SGD_Regressor"):
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv = 5)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
        (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
        alpha = best.alpha
        l1_ratio = best.l1_ratio
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train.reshape(-1))
        y_pred = lr.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
        (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        predictions = lr.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(lr, "model", signature=signature)

    params = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5]}
    with mlflow.start_run(run_name="Decision_Tree"):
        lr = DecisionTreeRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv = 5)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1,1))
        (rmse, mae, r2)  = eval_metrics(power_trans.inverse_transform(y_val), y_price_pred)
        max_depth = best.max_depth
        min_samples_split = best.min_samples_split
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
            
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
    
    current_experiment = mlflow.get_experiment_by_name("linear model students' math score")
    exp_id = current_experiment.experiment_id
    dfruns = mlflow.search_runs()
    best_model = mlflow.search_logged_models(
        experiment_ids=[exp_id],
        max_results=1,
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        output_format="list",
    )[0]
    best_run = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]
    run_id = best_run.run_id
    output_dir = os.path.join(os.getcwd(), "best_model_dir")
    path2model = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model",dst_path=output_dir)
    print(path2model)
