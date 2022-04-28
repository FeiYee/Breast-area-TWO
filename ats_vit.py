from __future__ import print_function

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import logging
import numpy as np

from vit_pytorch.ats_vit import ViT


def get_model(device):

    model = ViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        max_tokens_per_depth = (256, 128, 64, 32, 16, 8), # a tuple that denotes the maximum number of tokens that any given layer should have. if the layer has greater than this amount, it will undergo adaptive token sampling
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    return model.to(device)

# set seed
seed = 2022

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = torch.device('cuda:1')

# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((256, 256)),
#         # transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )


# test_transforms = transforms.Compose(
#     [
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ]
# )

# train_data = datasets.ImageFolder('new_data/training_dataset/', transform=train_transforms)
# test_data = datasets.ImageFolder('new_data/training_dataset/', transform=test_transforms)


# from sklearn.model_selection import KFold

# data_induce = np.arange(0, 7909)
# kf = KFold(n_splits=5)
# batch_size = 64

# fold_index = 0

# for train_index, val_index in kf.split(data_induce):

#     logging.basicConfig(level=logging.INFO,
#                     filename = "log_BreaKHis_v1/vit_ats.log",
#                     format='[%(asctime)s] - %(message)s')
#     fold_index += 1
#     train_subset = torch.utils.data.dataset.Subset(train_data, train_index)
#     val_subset = torch.utils.data.dataset.Subset(train_data, val_index)
#     train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    
#     logging.info('Start the training of ' + str(fold_index) + 'th fold')


train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

train_data = datasets.ImageFolder('new_data/training_dataset/', transform=train_transforms)
test_data = datasets.ImageFolder('new_data/training_dataset/', transform=test_transforms)


# from sklearn.model_selection import KFold

data_induce = np.arange(0, 7909)
np.random.shuffle(data_induce)
fold_num = 5
# kf = KFold(n_splits=fold_num)
batch_size = 64

fold_index = 0

for fold_index in range(fold_num):

    logging.basicConfig(level=logging.INFO,
                    filename = "log_BreaKHis_v1/vit_ats.log",
                    format='[%(asctime)s] - %(message)s')
    

    if fold_index == fold_num - 1:
        val_index = \
            data_induce[len(data_induce) // fold_num * fold_index: ]
    val_index = \
        data_induce[len(data_induce) // fold_num * fold_index: len(data_induce) // fold_num * (fold_index+1)]
    train_index = []
    for i in data_induce:
        if not i in val_index:
            train_index.append(i)

    fold_index += 1

    train_subset = torch.utils.data.dataset.Subset(train_data, train_index)
    val_subset = torch.utils.data.dataset.Subset(train_data, val_index)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    
    logging.info('Start the training of ' + str(fold_index) + 'th fold')

    # train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

    model = get_model(device)

    epochs = 100
    lr = 2e-5
    gamma = 0.7

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 11 * 20,gamma = 0.5)

    accs = []
    best_acc = 0

    criterion_test = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        # epoch_accuracy = 0
        train_acc_num, train_tot_num = 0, 0
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # acc = (output.argmax(dim=1) == label).float().mean()
            # epoch_accuracy += acc / len(train_loader)
            

            train_acc_num += (output.argmax(dim=1) == label).sum()
            train_tot_num += len(data)

            epoch_loss += loss / len(train_loader)
            epoch_accuracy_train = train_acc_num/ train_tot_num

        with torch.no_grad():
            epoch_test_accuracy = 0
            epoch_test_loss = 0
            test_acc_num, test_tot_num = 0, 0
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)

                test_output = model(data)
                test_loss = criterion_test(test_output, label)

                test_acc_num += (test_output.argmax(dim=1) == label).sum()
                test_tot_num += len(data)
                # print(test_acc_num)
                # print(test_tot_num)
                epoch_accuracy_test = test_acc_num/ test_tot_num
                # epoch_test_accuracy += acc / len(test_loader)
                epoch_test_loss += test_loss / len(test_loader)

        if epoch_accuracy_test > best_acc:
            print(epoch_accuracy_test)
            print(best_acc)
            best_acc = epoch_accuracy_test
            torch.save(obj=model.state_dict(), f='model_BreaKHis_v1/best_vit_ats_' + str(fold_index) +'.pth')

        
        logging.info(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.6f} - acc: {epoch_accuracy_train:.6f} - test_loss : {epoch_test_loss:.6f} - test_acc: {epoch_accuracy_test:.6f}\n"
        )
    
    logging.info(
        "best_acc of " + str(fold_index) + 'th fold is :'
    )

    logging.info(
        best_acc
    )

    logging.info(
        '-'*100
    )
