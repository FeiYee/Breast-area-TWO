import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
# from vit_pytorch.ats_vit import ViT
from vit_pytorch import ViT
import logging
import numpy as np
import random
import os


# def get_model(device):

#     model = ViT(
#         image_size = 256,
#         patch_size = 16,
#         num_classes = 2,
#         dim = 1024,
#         depth = 6,
#         max_tokens_per_depth = (256, 128, 64, 32, 16, 8), # a tuple that denotes the maximum number of tokens that any given layer should have. if the layer has greater than this amount, it will undergo adaptive token sampling
#         heads = 16,
#         mlp_dim = 2048,
#         dropout = 0.1,
#         emb_dropout = 0.1
#     )

#     return model.to(device)

def get_model(device):

    model = ViT(
        channels = 3,
        image_size = 256,
        patch_size = 32,
        num_classes = 2,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    return model.to(device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':

    seed_everything(2022)

    test_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)
    # test_data = datasets.ImageFolder('Dataset_BUSI_with_GT/train', transform=test_transforms)
    train_data = datasets.ImageFolder('new_data/training_dataset/', transform=test_transforms)

    device = torch.device('cuda:3')

#     model = ViT(
#     image_size = 256,
#     patch_size = 16,
#     num_classes = 2,
#     dim = 1024,
#     depth = 6,
#     max_tokens_per_depth = (256, 128, 64, 32, 16, 8), # a tuple that denotes the maximum number of tokens that any given layer should have. if the layer has greater than this amount, it will undergo adaptive token sampling
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

    model = get_model(device) 

    data_induce = np.arange(0, 7909)
    np.random.shuffle(data_induce)
    fold_num = 5
    batch_size = 64
    fold_index = 0

    for fold_index in range(fold_num):

        logging.basicConfig(level=logging.INFO,
                        filename = "log_BreaKHis_v1/base_eval.log",
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

        # train_subset = torch.utils.data.dataset.Subset(train_data, train_index)
        val_subset = torch.utils.data.dataset.Subset(train_data, val_index)
        # train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)
        
        logging.info('Start the eval of ' + str(fold_index) + 'th fold')
        logging.info('model_BreaKHis_v1/best_base_' + str(fold_index) +'.pth')


        model.load_state_dict(torch.load('model_BreaKHis_v1/best_base_' + str(fold_index) +'.pth'))
        #  model.load_state_dict(torch.load('./model/vit_ats.pth'))
        model.to(device)
        model.eval()


        correct_sum = 0.0
        total = 0

        # Initialize the prediction and label lists(tensors)
        preds = torch.zeros(0, dtype=torch.long, device=device)
        labels = torch.zeros(0, dtype=torch.long, device=device)

        # label_names = ["call","fenxin","normal","smoke","tired"]
        for n_iter, (image, label) in enumerate(test_loader):

            image = image.to(device)
            label = label.to(device)
            output = model(image)

            labels = torch.cat((labels, label), 0)
            preds = torch.cat((preds, output.argmax(dim=1)), 0)

            correct = output.argmax(dim=1) == label
            correct_sum += correct.sum()


        report = classification_report(labels.cpu().numpy(), preds.cpu().numpy(),
                                    digits=6)
    
        print("Top 1 err: {:.3}%".format((1 - correct_sum / len(test_loader.dataset))*100))
        print(report)
        logging.info(report)

        # print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
