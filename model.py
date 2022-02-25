import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import tensorflow as tf
import numpy as np

from datetime import datetime
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from build_features import build_features


def train_model(model, criterion, optimizer, scheduler, num_epochs, data_dir, batch_size, output_dir):
    """Train the model."""

    # load training/validation data
    dataloaders, dataset_sizes, class_names = build_features(data_dir, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log_dir = output_dir
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                time.sleep(0.1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            with train_summary_writer.as_default() :
                tf.summary.scalar(f'{phase}_loss', np.float(epoch_loss), step=epoch)
                tf.summary.scalar(f'{phase}_acc', np.float(epoch_acc), step=epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Новая лучшая модель!')
                torch.save(model, f'{output_dir}/checkpoints/best.pt')
            torch.save(model, f'{output_dir}/checkpoints/epoch_{epoch}.pt')


            # # log the best val accuracy to AML run
            # run.log('best_val_acc', np.float(best_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, np.float(best_acc)


def fine_tune_model(num_epochs, data_dir, learning_rate, momentum, num_classes, batch_size, output_dir, num_mod):
    """Load a pretrained model and reset the final fully connected layer."""

    if num_mod == 1:
        model_ft = models.resnet18(pretrained=True)
    elif num_mod == 2:
        model_ft = models.resnet34(pretrained=True)
    elif num_mod == 3:
        model_ft = models.resnet50(pretrained=True)
    elif num_mod == 4:
        model_ft = models.resnet101(pretrained=True)
    elif num_mod == 5:
        model_ft = models.resnet152(pretrained=True)
    elif num_mod == 6:
        model_ft = models.efficientnet_b0(pretrained=True)
    elif num_mod == 7:
        model_ft = models.efficientnet_b1(pretrained=True)
    elif num_mod == 8:
        model_ft = models.efficientnet_b2(pretrained=True)
    elif num_mod == 9:
        model_ft = models.efficientnet_b3(pretrained=True)
    elif num_mod == 10:
        model_ft = models.efficientnet_b4(pretrained=True)
    elif num_mod == 11:
        model_ft = models.efficientnet_b5(pretrained=True)
    elif num_mod == 12:
        model_ft = models.efficientnet_b6(pretrained=True)
    elif num_mod == 13:
        model_ft = models.efficientnet_b7(pretrained=True)

    if num_mod in [1, 2, 3, 4, 5]:
        num_ftrs = model_ft.fc.in_features
    elif num_mod in [6, 7, 8, 9, 10, 11, 12, 13]:
        num_ftrs = 1280

    model_ft.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)  # classes to predict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(),
                             lr=learning_rate, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    print('Началось обучение модели')
    model, best_acc = train_model(model_ft, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs, data_dir, batch_size, output_dir)
    print('Закончилось обучение модели')

    return model, best_acc