#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings


warnings.filterwarnings('ignore')

# In[2]:


use_gpu=torch.cuda.is_available()


# In[3]:


transform=transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


# In[7]:


image_path='/home/dm/pytorchLearn/NNData/DogVsCat'
data_image={x:datasets.ImageFolder(root=os.path.join(image_path,x),transform=transform) for x in ['train','val']}
data_loader_image={x:torch.utils.data.DataLoader(dataset=data_image[x], shuffle=True, batch_size=32) for x in ['train', 'val']}


# In[32]:


classes=data_image['train'].classes
classes_index=data_image['train'].class_to_idx


# In[34]:


model=models.resnet18(pretrained=True)


# In[37]:


for param in model.parameters():
    param.requires_grad=False


# In[38]:


num_in_ftrs=model.fc.in_features
model.fc=nn.Linear(num_in_ftrs,2)


# In[39]:


if use_gpu:
    model=model.cuda()


# In[40]:


Epochs=100
lr=0.001
BatchSize=32


# In[41]:


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters())


# In[44]:




for epoch in range(Epochs):
    print('epoch{}/{}'.format(epoch,Epochs))
    print('-'*10)
    
    for param in ['train','val']:
        if param=='train':
            model.train=True
        else:
            model.train=False

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
                print('Batch:{} Loss:{:.4f} Train Acc:{:.4f}%'.format(batch_index,running_loss/(batch_index),100*running_correct/(batch_index*BatchSize)))

            
        epoch_loss=running_loss/batch_index
        epoch_correct=100*running_correct/len(data_image[param])
        
        print('{} Loss:{:.4f} {} Acc:{:.4f}%'.format(param,epoch_loss,param,epoch_correct))
        


# In[ ]:




