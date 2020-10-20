import os
import subprocess
import sys
import warnings

import mlflow

warnings.filterwarnings('ignore')

PROJECT_DIR = sys.path[0]
os.chdir(PROJECT_DIR)

experiment_name = 'rented_bikes'
mlflow.set_experiment(experiment_name)

PORT = 5001  # REST API serving port
CONTAINER_NAME = "mlflow_example_model_serving"

best_run_df = mlflow.search_runs(order_by=['metrics.RMSE_CV ASC'], max_results=1)
if len(best_run_df.index) == 0:
    raise Exception(f"Found no runs for experiment '{experiment_name}'")

best_run = mlflow.get_run(best_run_df.at[0, 'run_id'])
best_model_uri = f"{best_run.info.artifact_uri}/model"
# best_model = mlflow.sklearn.load_model(best_model_uri)

# print best run info
print("Best run info:")
print(f"Run id: {best_run.info.run_id}")
print(f"Run parameters: {best_run.data.params}")
print("Run score: RMSE_CV = {:.4f}".format(best_run.data.metrics['RMSE_CV']))
print(f"Run model URI: {best_model_uri}")

# remove current container if exists
subprocess.run(f"docker rm --force {CONTAINER_NAME}", shell=True, check=False, stdout=subprocess.DEVNULL)

# run mlflow model serving in a docker container
docker_run_cmd = f"""
docker run
--name={CONTAINER_NAME}
--volume={PROJECT_DIR}:{PROJECT_DIR}
--publish {PORT}:{PORT}
--interactive
--rm
mlflow_example
mlflow models serve --model-uri {best_model_uri} --host 0.0.0.0 --port {PORT} --workers 2 --no-conda
""".replace('\n', ' ').strip()
print(f"Running command:\n{docker_run_cmd}")

subprocess.run(docker_run_cmd, shell=True, check=True)
