#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
import torchvision
from torchvision import transforms, datasets


# In[20]:


norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)

batch_size = 32
validation_batch_size = 32

data_dir = 'D:/AI_Hackatlon/AI Vision Challenge Dataset/hackathon_dataset'

transform_train = transforms.Compose([transforms.ToTensor()] )
dataset = torchvision.datasets.ImageFolder(root= data_dir, transform = transform_train)


# In[ ]:


classes = os.listdir(data_dir)
print(classes)


# In[21]:


# normalization values for pretrained ResNet18 

norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)

batch_size = 32
validation_batch_size = 32



# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomRotation(degrees=15),
                                      transforms.ColorJitter(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean, norm_std)])

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean,norm_std)])

validation_transforms = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean,norm_std)])


# In[ ]:


import torch as th
import math
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

#Loading in the dataset
data = {
    'train':
    datasets.ImageFolder(root=data_dir, transform=train_transforms),
    'valid':
    datasets.ImageFolder(root=data_dir, transform=validation_transforms),
    'test':
    datasets.ImageFolder(root=data_dir, transform=test_transforms),
}

train_data = datasets.ImageFolder(data_dir)
# number of subprocesses to use for data loading
num_workers = 0
# percentage of training set to use as validation and test 
valid_size = 0.2

test_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
valid_split = int(np.floor((valid_size) * num_train))
test_split = int(np.floor((valid_size+test_size) * num_train))
valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

train_idx_2000 = list(train_idx) 
valid_idx_500 = list(valid_idx)
test_idx_800 = list(test_idx)

print(len(train_idx), len(valid_idx), len(test_idx))
print(len(train_idx_2000), len(valid_idx_500), len(test_idx_800))

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx_2000)
valid_sampler = SubsetRandomSampler(valid_idx_500)
test_sampler = SubsetRandomSampler(test_idx_800)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(data['train'], batch_size=32,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(data['valid'], batch_size=32, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(data['test'], batch_size=32, 
    sampler=test_sampler, num_workers=num_workers)


# # USING PRE-TRAINED RESNET18

# In[23]:


def get_accuracy(predicted, labels):
    batch_len, correct= 0, 0
    batch_len = labels.size(0)
    correct = (predicted == labels).sum().item()
    return batch_len, correct
def evaluate(model, val_loader):
    losses= 0
    num_samples_total=0
    correct_total=0
    model.eval()
    for inputs, labels in val_loader:
       
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        loss = criterion(out, labels)
        losses += loss.item() 
        b_len, corr = get_accuracy(predicted, labels)
        num_samples_total +=b_len
        correct_total +=corr
    accuracy = correct_total/num_samples_total
    losses = losses/len(val_loader)
    return losses, accuracy


# In[24]:


resnet_18 = torch.load("modelRes185.pth")


# In[25]:


#On test set:
total = 0
correct = 0
resnet_18.eval()
k = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        
        outputs = resnet_18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        k += labels.size(0)

print(k)        
print('Accuracy of the network on the test images: %2f %%' % (
    100 * correct / total))


# In[26]:


#On test class by class:
class_correct = list(0. for i in range(len(classes)))
class_total = list(1e-7 for i in range(len(classes)))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        
        outputs = resnet_18(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(3):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of %5s : %2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[3]:


################################# FOR YOUR TEST ##############################################################################
#################  JUST CHANGE "data_dirtest" FOLDER LINK

import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

def get_accuracy(predicted, labels):
    batch_len, correct= 0, 0
    batch_len = labels.size(0)
    correct = (predicted == labels).sum().item()
    return batch_len, correct
def evaluate(model, val_loader):
    losses= 0
    num_samples_total=0
    correct_total=0
    model.eval()
    for inputs, labels in val_loader:
       
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        loss = criterion(out, labels)
        losses += loss.item() 
        b_len, corr = get_accuracy(predicted, labels)
        num_samples_total +=b_len
        correct_total +=corr
    accuracy = correct_total/num_samples_total
    losses = losses/len(val_loader)
    return losses, accuracy

resnet_18 = torch.load("modelRes185.pth")

norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)


test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(norm_mean,norm_std)])


############################### CHANGE THIS LINK >>>>>>>>>>>>>>>>
data_dirtest = 'D:/AI_Hackatlon/AI Vision Challenge Dataset/hackathon_dataset'


test_data = datasets.ImageFolder(data_dirtest)
data_test = datasets.ImageFolder(root=data_dirtest, transform=test_transforms)
num_test = len(test_data)
indices = list(range(num_test))
test_sampler = SubsetRandomSampler(indices)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=32, 
    sampler=test_sampler, num_workers=0)

#On test set:
total = 0
correct = 0
resnet_18.eval()
k = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        
        outputs = resnet_18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        k += labels.size(0)

print(k)        
print('Accuracy of the network on the test images: %2f %%' % (
    100 * correct / total))


# In[ ]:




