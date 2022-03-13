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


https://colab.research.google.com/drive/1jDd6D0fvfcJ128uCqG2OVHUfJMFjuN6z?usp=sharing - инференс (веса best.pt - resnet152)

## Работающий прототип с веб камерой
camera.ipynb

классификатор из пунтка 1. + детектор из opencv

Два режима: 
- фото - снимок с камеры с детектированием лица и определелением эмоции
- видео - детектирование и определение лица в режиме видео онлайн
