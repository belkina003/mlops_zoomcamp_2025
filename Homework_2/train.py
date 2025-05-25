import os
import pickle
import click
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("experiment-hw2")


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    with mlflow.start_run():
        mlflow.set_tag("developer", "Olga")

        mlflow.log_param("train-data-path","./data/green_tripdata_2023-01.parquet")
        mlflow.log_param("val-data-path","./data/green_tripdata_2023-02.parquet")

        max_depth=10
        mlflow.log_param("max_depth", max_depth)
        rf = RandomForestRegressor(max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("min_samples_split", rf.get_params()['min_samples_split'])
        mlflow.log_artifact(local_path="mlflow.db", artifact_path="models_pickle/")



if __name__ == '__main__':
    run_train()