import mlflow
import os

MLFLOW_SERVER_URL = 'http://localhost:11114/'

# подключаемся к серверу
mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
experiment_name = 'ml_diplom'
mlflow.set_experiment(experiment_name)

with mlflow.start_run() :
    mlflow.log_params(
        {   "name" : '20220223-193729',
            "batch_size" : 64,
            "model_name" : 'resnet18',
            "optim" : 'sgd',
            "learning_rate" : 0.001,
            "momentum" : 0.9,
            "epochs" : 15,
            "augm" : 'RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAutocontrast'
        }
    )

    mlflow.log_metric("res_pr", 0.41360)
    mlflow.log_metric("res_pu", 0.43000)

# RandomResizedCrop
# RandomHorizontalFlip
# RandomRotation
# ColorJitter
# RandomAutocontrast