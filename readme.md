# Система по распознаванию эмоций
## 1. Модель классификации эмоций

train.py arg - запуск обучения моделей
arg - номера моделей через запятую

1 - resnet18

2 - resnet34

3 - resnet50

4 - resnet101

5 - resnet152

6 - efficientnet_b0

7 - efficientnet_b1

8 - efficientnet_b2

9 - efficientnet_b3

10 - efficientnet_b4

11 - efficientnet_b5

12 - efficientnet_b6

13 - efficientnet_b7



predict.py - предсказания на тесте + формирование файла csv

inference.py - инференс модели

https://colab.research.google.com/drive/1jDd6D0fvfcJ128uCqG2OVHUfJMFjuN6z?usp=sharing - инференс модели

'./result/resnet50_2022_03_15-13_31/checkpoints/best.pt' - лучшие веса на паблик скор (best_public.pt)

'./result/resnet152_2022_02_24-18_36/checkpoints/best.pt'  - лучшие веса на прайват скор (best_private.pt)

## 2. Работающий прототип с веб камерой (детектирование лица + классификация эмоции)

camera.ipynb

классификатор из пунтка 1. + детектор из opencv

Два режима: 
- фото - снимок с камеры с детектированием лица и определелением эмоции
- видео - детектирование и определение эмоции в режиме видео онлайн с камеры
