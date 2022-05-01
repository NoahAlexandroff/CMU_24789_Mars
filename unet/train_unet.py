import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ai4mars_dataset import AI4MarsDataset
from unet import UNet
import pickle
from unet_loss import combined_loss
import argparse
import torchmetrics

def compute_confusion_matrix(list_cms):
    stacked_cms = torch.stack((list_cms), dim=0)
    sum_rows_cms = torch.sum(stacked_cms, dim=0)
    percentage_cm = torch.sum(stacked_cms, dim=0)/torch.sum(sum_rows_cms, dim=1).unsqueeze(1) 
    return percentage_cm


def main(num_epochs=10, batch_size=16, dataroot="./data/msl/", image_size = 256, train_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        num_workers = 0
    else:
        torch.cuda.empty_cache()
        num_workers = 2

    train_dataset = AI4MarsDataset(folder_path=dataroot, is_train=True, image_size=image_size)
    test_dataset = AI4MarsDataset(folder_path=dataroot, is_train=False, image_size=image_size)

    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    model = UNet(in_channels=1, n_classes=5)
    record = {"train_size":train_size,
              "training_acc":[], "test_acc":[], "training_loss":[], "test_loss":[],
              "training_dice":[], "test_dice":[], "training_jaccard":[], "test_jaccard":[],
              "training_cm":[], "test_cm":[]}


    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    best_model_test_acc = 0
    
    for epoch in range(num_epochs):  # loop over the dataset multiple timesi
        
        train_loss, test_loss = 0.0, 0.0
        train_acc, test_acc = 0.0, 0.0
        train_dice, test_dice = 0.0, 0.0 
        train_jaccard, test_jaccard = 0.0, 0.0 
        train_cm, test_cm = [], []
        model.train()
        
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            images, depths, labels = data
            num_in_batch = images.shape[0]
            images=images.reshape(num_in_batch,1,image_size,image_size)
            depths=depths.reshape(num_in_batch,1,image_size,image_size)
            labels = labels.reshape(num_in_batch,image_size,image_size).long()
            labels[labels>=255]=4
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            outputs = model(images.to(device))
            loss = combined_loss(labels.to(device),outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            train_acc += torchmetrics.functional.accuracy(outputs, labels.to(device)).item()
            train_dice += torchmetrics.functional.dice_score(outputs, labels.to(device)).item()
            train_jaccard += torchmetrics.functional.jaccard_index(outputs, labels.to(device), num_classes=5).item()
            train_cm.append(torchmetrics.functional.confusion_matrix(outputs, labels.to(device), num_classes=5))

        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, depths, labels = data
                num_in_batch = images.shape[0]
                images=images.reshape(num_in_batch,1,image_size,image_size)
                depths=depths.reshape(num_in_batch,1,image_size,image_size)
                labels = labels.reshape(num_in_batch,image_size,image_size).long()
                labels[labels==255]=4
                outputs = model(images.to(device))
                loss = combined_loss(labels.to(device),outputs)
                test_loss += loss.item()
                test_acc += torchmetrics.functional.accuracy(outputs, labels.to(device)).item()
                test_dice += torchmetrics.functional.dice_score(outputs, labels.to(device)).item()
                test_jaccard += torchmetrics.functional.jaccard_index(outputs, labels.to(device), num_classes=5).item()
                test_cm.append(torchmetrics.functional.confusion_matrix(outputs, labels.to(device), num_classes=5))

        record["training_acc"].append(train_acc/len(trainloader))
        record["test_acc"].append(test_acc/len(testloader))
        record["training_loss"].append(train_loss/len(trainloader))
        record["test_loss"].append(test_loss/len(testloader))
        record["training_dice"].append(train_dice/len(trainloader))
        record["test_dice"].append(test_dice/len(testloader))
        record["training_jaccard"].append(train_jaccard/len(trainloader))
        record["test_jaccard"].append(test_jaccard/len(testloader))
        record["training_cm"].append(compute_confusion_matrix(train_cm))
        record["test_cm"].append(compute_confusion_matrix(train_cm))

        with open("./record.pkl", "wb") as fp:   # Unpickling
            pickle.dump(record, fp)

        print('Epoch %d| Train Acc: %.4f| Test Acc: %.3f| Train Loss: %.3f| Test Loss: %.3f| Train Dice: %.3f| Test Dice: %.3f| Train Jaccard: %.3f| Test Jaccard: %.3f|'%(
            epoch+1, record["training_acc"][epoch], record["test_acc"][epoch],
            record["training_loss"][epoch], record["test_loss"][epoch],
            record["training_dice"][epoch], record["test_dice"][epoch],
            record["training_jaccard"][epoch], record["test_jaccard"][epoch]))

        if record["test_jaccard"][epoch]>best_model_test_acc:
            best_model_test_acc=record["test_jaccard"][epoch]
            torch.save(model.state_dict(), './model.pth')

    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', help='The number of datapoints to train with.',
                        default=256)
    args = parser.parse_args()
    main(train_size=args.train_size)