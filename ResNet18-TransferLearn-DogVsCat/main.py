#!/usr/bin/env python
# coding: utf-8

# In[16]:
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import argparse
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser(description='transfer learning for DogVsCat problem with ResNet18')

parser.add_argument('--lr','--learning-rate',default=0.001,type=float,help='initial learning rate')

parser.add_argument('--batch-size',default=32,type=int)

parser.add_argument('--epoch-num',default=100,type=int)

parser.add_argument('--resume',default='',type=str)

parser.add_argument('--start-epoch',default=0,type=int)


def save_checkpoint(state,is_best_model,file_name='checkpoint.pth.tar'):
    torch.save(state,file_name)
    if is_best_model:
        shutil.copyfile(file_name,'best_model.pth.tar')


def main():
    args=parser.parse_args()

    best_acc=0.0

    use_gpu=torch.cuda.is_available()

    transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])


    image_path='/home/dm/pytorchLearn/NNData/DogVsCat'  
    data_image={x:datasets.ImageFolder(root=os.path.join(image_path,x),transform=transform) for x in ['train','val']}
    data_loader_image={x:torch.utils.data.DataLoader(dataset=data_image[x], shuffle=True, batch_size=32) for x in ['train', 'val']}


    classes=data_image['train'].classes
    classes_index=data_image['train'].class_to_idx


    model=models.resnet18(pretrained=True)

    # if use_gpu:
    #     model=model.cuda()

    for param in model.parameters():
        param.requires_grad=False

    num_in_ftrs=model.fc.in_features
    model.fc=nn.Linear(num_in_ftrs,2)

    #model = torch.nn.DataParallel(model).cuda()
    model=model.cuda()

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint in {}'.format(args.resume))

            checkpoint=torch.load(args.resume)
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            best_acc=checkpoint['best_acc']
            args.start_epoch=checkpoint['epoch']

        else:
            print('no found file of {}'.format(args.resume))






    for epoch in range(args.start_epoch,args.epoch_num):
        print('\nepoch{}/{} learning-rate {} batch-size {}'.format(epoch, args.epoch_num, args.lr, args.batch_size))
        print('-'*10)
        
        for param in ['train','val']:
            if param=='train':
                #model.train=True
                model.train()
            else:
                #model.train=False
                model.eval()

            running_loss=0.0
            running_correct=0
            
            batch_index=0
            
            for data in data_loader_image[param]:
                batch_index+=1
                
                X,y=data
                if use_gpu:
                    X,y=Variable(X.cuda()),Variable(y.cuda())
                else:
                    X,y=Variable(X),Variable(y)
                    
                optimizer.zero_grad()
                y_pred=model(X)
                _,pred=torch.max(y_pred.data,1)
                
                loss=criterion(y_pred,y)
                
                if param == 'train':
                    loss.backward()
                    optimizer.step()
                    
                running_loss+=loss
                running_correct+=torch.sum(pred==y.data)
                
                if batch_index%100 == 0 and param == 'train':
                    print('Batch:{} Loss:{:.4f} Train Acc:{:.4f}%'.format(batch_index,running_loss/(batch_index),100*float(running_correct)/(batch_index*args.batch_size)))

                
            epoch_loss=running_loss/batch_index
            epoch_acc=float(running_correct)/len(data_image[param])
            

            print('{} Loss:{:.4f} {} Acc:{:.4f}%'.format(param,epoch_loss,param,100*epoch_acc))

            if param=='val':
                is_best_model = epoch_acc>best_acc

                best_acc=max(epoch_acc,best_acc)

                save_checkpoint({
                    'epoch':epoch+1,
                    'state_dict':model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                },is_best_model)


if __name__ == '__main__':
    main()

        






