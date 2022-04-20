from re import A
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from ai4mars_dataset import AI4MarsDataset
from unet import UNet
import pickle


def main(num_epochs=5, batch_size=4, dataroot="./data/msl/", image_size = 256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        num_workers = 0
    else:
        torch.cuda.empty_cache()
        num_workers = 4
    # load and transform dataset

    train_dataset = AI4MarsDataset(folder_path=dataroot, is_train=True)
    test_dataset = AI4MarsDataset(folder_path=dataroot, is_train=False)

    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
    
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)



    model = UNet(in_channels=1, n_classes=5)
    record = [[], [], []]


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    best_model_test_acc = 0
    
    for epoch in range(num_epochs):  # loop over the dataset multiple timesi
        
        running_loss = 0.0
        train_total, test_total = 0, 0
        train_correct, test_correct = 0, 0
        model.train()
        
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            images, depths, labels = data
            images=images.reshape(batch_size,1,image_size,image_size)
            depths=depths.reshape(batch_size,1,image_size,image_size)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            
            outputs = model(images.to(device))
            
            # outputs = outputs.reshape(batch_size,5,image_size,image_size)
            labels = labels.reshape(batch_size,image_size,image_size).long()
            labels[labels==255]=4
            print(outputs.shape)
            print(labels.shape)
            
            loss = criterion(outputs, labels.to(device))
            print('loss_in')
            loss.backward()
            print('loss_out')
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            seg_acc = (labels.cpu() == torch.argmax(outputs, axis=1).cpu()).sum() / torch.numel(labels.cpu()).item()
            print(seg_acc)
            _, train_predict = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (train_predict == labels.to(device)).sum().item()
            print(loss.item())
    
        with torch.no_grad():
            model.eval()
            for data in testloader:
                images, depths, labels = data
                images=images.reshape(batch_size,1,image_size,image_size)
                depths=depths.reshape(batch_size,1,image_size,image_size)
                outputs = model(images.to(device))
                _, test_predict = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (test_predict == labels.to(device)).sum().item()
        
        train_acc = train_correct/train_total
        test_acc = test_correct/test_total
        record[0].append(train_acc)
        record[1].append(test_acc)
        record[2].append(running_loss/len(trainloader))
        print('Epoch %d| Train loss: %.4f| Train Acc: %.3f| Test Acc: %.3f'%(
            epoch+1, running_loss/len(trainloader), train_acc, test_acc))
        if test_acc>best_model_test_acc:
            best_model_test_acc=test_acc
            
            torch.save(model.state_dict(), PATH)
            with open("./record.pkl", "wb") as fp:   # Unpickling
                pickle.dump(record, fp)

    print('Finished Training')
  

    return record


if __name__ == "__main__":
    main()