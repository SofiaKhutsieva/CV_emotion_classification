import os
import mlflow
import sys
import re

from model import fine_tune_model
from datetime import datetime

# НОМЕР МОДЕЛИ
num_mods_ = str(sys.argv[1])
num_mods = re.findall('\d+', num_mods_)

dict_model = {1: 'resnet18',
              2: 'resnet34',
              3: 'resnet50',
              4: 'resnet101',
              5: 'resnet152',
              6: 'efficientnet_b0',
              7: 'efficientnet_b1',
              8: 'efficientnet_b2',
              9: 'efficientnet_b3',
              10: 'efficientnet_b4',
              11: 'efficientnet_b5',
              12: 'efficientnet_b6',
              13: 'efficientnet_b7'
              }
# ПАРАМЕТРЫ
# данные
data_dir = './data'
batch_size = 64

# обучение
num_epochs = 50
learning_rate = 0.001
momentum = 0.9
num_classes = len(os.listdir('./data/train'))
print(f'num_classes = {num_classes}')


# МЛФЛОУ
MLFLOW_SERVER_URL = 'http://localhost:11114/'

# подключаемся к серверу
mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
experiment_name = 'ml_diplom_new'
mlflow.set_experiment(experiment_name)

# ОБУЧЕНИЕ
for num_mod in num_mods:
    num_mod = int(num_mod)
    print(f'num_mod {num_mod}')

    output_dir = f'./result/{dict_model[num_mod]}_{datetime.now().strftime("%Y_%m_%d-%H_%M")}'
    os.mkdir(f'{output_dir}')
    os.mkdir(f'{output_dir}/checkpoints')
    os.mkdir(f'{output_dir}/submission_file')

    model, best_acc = fine_tune_model(num_epochs, data_dir, learning_rate, momentum, num_classes, batch_size, output_dir, num_mod)

    with mlflow.start_run() :
        mlflow.log_params(
            {"output_dir": output_dir,
             "batch_size": batch_size,
             "model_name": dict_model[num_mod],
             "optim": 'sgd',
             "learning_rate": learning_rate,
             "momentum": momentum,
             "augm": 'RandomResizedCrop, RandomAutocontrast'
            }
        )

        mlflow.log_metric("best_acc", best_acc)