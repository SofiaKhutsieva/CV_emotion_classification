import torch
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from build_features import build_features_predict

directory = './data/test'
model_dir = './result/resnet152_2022_02_24-18_36/checkpoints/epoch_13.pt' #
df_dir = f'./result/{model_dir.split("/")[-3]}/submission_file/{model_dir.split("/")[-3]}_{model_dir.split("/")[-1][:-3]}.csv'

emotions, image_paths = [], []
for index, pred_dir in enumerate(os.listdir(directory)):
    print(index)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_dir)
    model.to(device)
    x = build_features_predict(f'{directory}/{pred_dir}')
    x = x.to(device)

    logits = model(x[None, ...])
    preds = tf.nn.softmax(logits.cpu().detach().numpy())
    pred = np.argmax(preds[0])
    dict = {0:'anger', 1:'contempt', 2:'disgust',
            3:'fear', 4:'happy', 5:'neutral', 6:'sad', 7:'surprise', 8:'uncertain'}
    emotions.append(dict[int(pred)])
    image_paths.append(pred_dir)
    # print(f'файл - {pred_dir}, предсказание - {dict[int(pred)]}, вероятность = {np.max(preds[0])}')

df = pd.DataFrame([])
df['image_path'] = image_paths
df['emotion'] = emotions
df.to_csv(df_dir, index=False)