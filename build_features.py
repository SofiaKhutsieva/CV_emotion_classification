import torch
import os

from torchvision import datasets, models, transforms
from PIL import Image


def build_features(data_dir, batch_size):
    """Load the train/val data."""

    # Data augmentation and normalization for training
    # Just normalization for validation
    print('Началось формирование признаков')
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), #отображение, ось - вертикаль
            # transforms.RandomRotation(degrees=(0, 180)), # повороты
            # transforms.ColorJitter(brightness=.5, hue=.3), # яркость, цвет
            # transforms.RandomAutocontrast(), #контрастность
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f'class_names: {class_names}')

    print('Закончилось формирование признаков')
    return dataloaders, dataset_sizes, class_names


def build_features_predict(pred_dir):

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = Image.open(pred_dir)
    image_features = data_transforms(img)

    return image_features