import itertools
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

# delete default experiment if exits
if mlflow.get_experiment_by_name("Default").lifecycle_stage == 'active':
    mlflow.delete_experiment("0")
subprocess.run("mlflow gc", shell=True, check=False, stdout=subprocess.DEVNULL)

# Model Hyper-parameters
parameters = {
    "learning_rate": [0.1, 0.05, 0.01],
    "max_depth": [4, 5, 6],
}

# Tuning the hyper-parameters via grid search
# generate parameters combinations
params_keys = parameters.keys()
params_values = [
    parameters[key] if isinstance(parameters[key], list) else [parameters[key]]
    for key in params_keys
]
runs_parameters = [
    dict(zip(params_keys, combination)) for combination in itertools.product(*params_values)
]

# execute experiment runs in parallel in docker containers
submitted_runs = []
for run_parameters in runs_parameters:
    submitted_runs.append(mlflow.projects.run(
        uri='.',
        backend='local',
        parameters=run_parameters,
        synchronous=False,
        docker_args={"user": f"{os.getuid()}:{os.getgid()}"},
    ))

print(f"Submitted {len(submitted_runs)} runs. Waiting for them to finish...")

# get runs status (blocking)
runs_status = [run.wait() for run in submitted_runs]

print(f"Experiment '{experiment_name}' finished!")
print(f"{sum(runs_status)} runs succeeded out of {len(runs_status)} submitted")
