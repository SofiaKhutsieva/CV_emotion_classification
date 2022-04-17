import torch
import tensorflow as tf
import numpy as np
import time

from build_features import build_features_predict

dict = {0 : 'anger', 1 : 'contempt', 2 : 'disgust',
        3 : 'fear', 4 : 'happy', 5 : 'neutral', 6 : 'sad', 7 : 'surprise', 8 : 'uncertain'}

def inference_image(images_dir, model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print(f'device {device}')
    model = torch.load(model_dir) # map_location='cpu'
    model.to(device)

    inference_times = []
    for image_dir in images_dir:
        start_time = time.time()
        x = build_features_predict(image_dir)
        x = x.to(device)
        logits = model(x[None, ...])
        preds = tf.nn.softmax(logits.cpu().detach().numpy())
        pred = np.argmax(preds[0])
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        print(f'для изображения {image_dir} эмоция - {dict[int(pred)]}, время инференса = {inference_time}')
    print(f'среднее время инференса = {sum(inference_times) / len(images_dir)}')

images_dir = ['./data/test/1.jpg', './data/test/2.jpg', './data/test/3.jpg', './data/test/4.jpg', './data/test/5.jpg',
              './data/test/6.jpg', './data/test/7.jpg', './data/test/8.jpg', './data/test/9.jpg', './data/test/10.jpg']
model_dir = './result/resnet50_2022_03_15-13_31/checkpoints/best.pt' # './result/resnet152_2022_02_24-18_36/checkpoints/epoch_13.pt' #

inference_image(images_dir, model_dir)