import torch
import tensorflow as tf
import numpy as np

from build_features import build_features_predict

dict = {0 : 'anger', 1 : 'contempt', 2 : 'disgust',
        3 : 'fear', 4 : 'happy', 5 : 'neutral', 6 : 'sad', 7 : 'surprise', 8 : 'uncertain'}

def inference_image(images_dir, model_dir):
    for image_dir in images_dir:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = torch.load(model_dir)
        model.to(device)
        x = build_features_predict(image_dir)
        x = x.to(device)

        logits = model(x[None, ...])
        preds = tf.nn.softmax(logits.cpu().detach().numpy())
        pred = np.argmax(preds[0])
        print(f'для изображения {image_dir} эмоция - {dict[int(pred)]}')

images_dir = ['./data/test/1.jpg', './data/test/2.jpg']
model_dir = './result/resnet152_2022_02_24-18_36/checkpoints/epoch_13.pt' #

inference_image(images_dir, model_dir)