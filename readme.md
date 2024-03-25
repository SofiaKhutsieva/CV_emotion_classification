# Система по распознаванию эмоций
## **Задача 1** 
Создать модель классификации эмоций  

```train.py arg``` - запуск обучения моделей  
arg - номера моделей через запятую

1 - resnet18,
2 - resnet34,
3 - resnet50,
4 - resnet101,
5 - resnet152,
6 - efficientnet_b0,
7 - efficientnet_b1,
8 - efficientnet_b2,
9 - efficientnet_b3,
10 - efficientnet_b4,
11 - efficientnet_b5,
12 - efficientnet_b6,
13 - efficientnet_b7


```predict.py``` - предсказания на тесте + формирование файла csv  
```inference.py``` - инференс модели  

https://colab.research.google.com/drive/1jDd6D0fvfcJ128uCqG2OVHUfJMFjuN6z?usp=sharing - инференс модели  
'./result/resnet50_2022_03_15-13_31/checkpoints/best.pt' - лучшие веса на паблик скор (best_public.pt)  
'./result/resnet152_2022_02_24-18_36/checkpoints/best.pt'  - лучшие веса на прайват скор (best_private.pt)

## **Задача 2**
Cоздать работающий прототип по детектированию лица и классификации эмоции, работающий с веб камеры в режиме фото и видео

```camera.ipynb``` - классификация эмоции из задачи 1 + детектирование лица из opencv

Два режима: 
- фото - снимок с камеры с детектированием лица и определелением эмоции
- видео - детектирование и определение эмоции в режиме видео онлайн с камеры

## **Результат**

**сравнение моделей классификации:**

![image](https://github.com/SofiaKhutsieva/CV_emotion_classification/assets/73535658/e0e7c421-e048-472d-b300-cb747d9ebc90)

**сравнение аугементаций:**

![image](https://github.com/SofiaKhutsieva/CV_emotion_classification/assets/73535658/90808a1e-f1a5-4e91-9227-2faea5462444)

**детектор + классификатор:**

![image](https://github.com/SofiaKhutsieva/CV_emotion_classification/assets/73535658/4bc16893-7cb4-4029-b645-5ca2539daeb7)

**итог:**

![image](https://github.com/SofiaKhutsieva/CV_emotion_classification/assets/73535658/45ce6325-5abd-42d3-8516-5cf6f3529fbd)

подробные результаты см. презентацию - https://github.com/SofiaKhutsieva/CV_emotion_classification/blob/main/%D0%BF%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F_%D1%85%D1%83%D1%86%D0%B8%D0%B5%D0%B2%D0%B0_%D1%81%D0%BE%D1%84%D0%B8%D1%8F.pptx

