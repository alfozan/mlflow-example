
# MLflow Example Project + Notebook

This repository provides an example of dataset preprocessing, model training and evaluation, model tuning and finally model serving (REST API) in a containerized environment using MLflow tracking, projects and models modules.

This project contains an MLflow project that trains a GBRT (Gradient Boosted Regression Tree) model on UC Irvine's [Bike Sharing Dataset Data Set](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) and uses a Docker image to capture the dependencies needed to run training and inference code. 

#### Talk: [@PyDataRiyadh](https://twitter.com/PyDataRiyadh): 
 - https://twitter.com/PyDataRiyadh/status/1291043529146466304
 - https://twitter.com/PyDataRiyadh/status/1314841078999154689

#### Talk slides: [MLflow-presentation.pdf](https://github.com/alfozan/mlflow-example/blob/master/MLflow-presentation.pdf "MLflow-presentation.pdf")

#### Notebook: [MLflow-example-notebook.ipynb](https://github.com/alfozan/mlflow-example/blob/master/MLflow-example-notebook.ipynb "MLflow-example-notebook.ipynb")


# Structure of this project

- [`MLproject`](https://github.com/alfozan/mlflow-example/blob/master/MLproject "MLproject") specifies the Docker container environment to run the project and defines `command` and `parameters` in `entry_points`  
- [`Dockerfile`](https://github.com/alfozan/mlflow-example/blob/master/Dockerfile "Dockerfile") used to build the image referenced by the `MLproject` 
- [`requirements.txt`](https://github.com/alfozan/mlflow-example/blob/master/requirements.txt "requirements.txt"): defined python dependencies needed to build training and inference docker image
- [`mlflow_project_driver.py`](https://github.com/alfozan/mlflow-example/blob/master/mlflow_project_driver.py "mlflow_project_driver.py"): creates an MLflow experiment for model training and tuning and launches MLflow runs in parallel in docker containers. 
- [`mlflow_model_driver.py`](https://github.com/alfozan/mlflow-example/blob/master/mlflow_model_driver.py "mlflow_model_driver.py"): finds best training run and starts a REST API model server based on [MLflow Models](https://www.mlflow.org/docs/latest/models.html) in docker containers. 
- [`train.py`](https://github.com/alfozan/mlflow-example/blob/master/train.py "train.py") : contains a file that trains a scikit-learn model and uses MLflow Tracking APIs to log the model and its metadata (e.g., hyperparameters and metrics)
- [`data/hour.csv`](https://github.com/alfozan/mlflow-example/blob/master/data/hour.csv "hour.csv"):  [Bike Sharing Dataset Data Set](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) 

# Getting started
Prerequisites: 

 1. Python 3
 2. Install Docker per instructions at https://docs.docker.com/install/overview/
 3. Install mlflow `pip install mlflow`

## Dockerized model training and training with MLflow:

 1. clone this repo: `git clone https://github.com/alfozan/mlflow-example`
 2. build the image for the project's Docker container environment: `docker build -t mlflow_example -f Dockerfile .`
 3. Start training and tracking: `python3 mlflow_project_driver.py`

## Run MLflow tracking UI:
In the same repo directory, run `mlflow ui --host 0.0.0.0 --port 5000`
UI is accessible at http://localhost:5000/


## Dockerized MLflow model serving (REST API)
In the same repo directory, run `python3 mlflow_model_driver.py`

## Inference request:
```bash
curl --silent --show-error 'http://localhost:5001/invocations' -H 'Content-Type: application/json' -d '{
    "columns": ["season", "year", "month", "hour_of_day", "is_holiday", "weekday", "is_workingday", "weather_situation", "temperature", "feels_like_temperature", "humidity", "windspeed"],
    "data": [[1, 0, 1, 0, 0, 6, 0, 1, 0.24, 0.2879, 0.81, 0.0000]]
}'
```
