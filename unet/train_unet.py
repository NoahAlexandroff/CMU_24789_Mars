import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ai4mars_dataset import AI4MarsDataset
from unet import UNet
import pickle
from unet_loss import dice_loss



def main(num_epochs=5, batch_size=2, dataroot="./data_subset/msl/", image_size = 256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        num_workers = 0
    else:
        torch.cuda.empty_cache()
        num_workers = 2
    # load and transform dataset

    train_dataset = AI4MarsDataset(folder_path=dataroot, is_train=True, image_size=image_size)
    test_dataset = AI4MarsDataset(folder_path=dataroot, is_train=False, image_size=image_size)

    train_size = int(256/256 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    model = UNet(in_channels=1, n_classes=5)
    record = [[], [], [], []]


    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    best_model_test_acc = 0
    
    for epoch in range(num_epochs):  # loop over the dataset multiple timesi
        
        training_loss, test_loss = 0.0, 0.0
        train_total, test_total = 0, 0
        train_correct, test_correct = 0, 0
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
            
            # loss = criterion(outputs, labels.to(device))
            loss = dice_loss(labels.to(device),outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()
            seg_acc = ((labels.cpu() == torch.argmax(outputs, axis=1).cpu()).sum() / torch.numel(labels.cpu())).item()
            print('batch_acc:',seg_acc)
            _, train_predict = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (train_predict == labels.to(device)).sum().item()
            print('loss:',loss.item())
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
                _, test_predict = torch.max(outputs.data, 1)
                loss = dice_loss(labels.to(device),outputs)
                test_total += labels.size(0)
                test_loss += loss.item()
                test_correct += (test_predict == labels.to(device)).sum().item()
        train_acc = train_correct/train_total
        test_acc = test_correct/test_total
        record[0].append(train_acc)
        record[1].append(test_acc)
        record[2].append(training_loss/len(trainloader))
        record[3].append(test_loss/len(testloader))

        print('Epoch %d| Train loss: %.4f| Train Acc: %.3f| Test Acc: %.3f'%(
            epoch+1, training_loss/len(trainloader), test_loss/len(testloader) , train_acc, test_acc))
        if test_acc>best_model_test_acc:
            best_model_test_acc=test_acc
            
            torch.save(model.state_dict(), './model.pth')
            with open("./record.pkl", "wb") as fp:   # Unpickling
                pickle.dump(record, fp)

    print('Finished Training')
  

    return record


if __name__ == "__main__":
    main()