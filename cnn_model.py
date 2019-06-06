#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        for parent, dirnames, filenames in os.walk(data_path):
            self.img_filename = [filenames for filenames in sorted(filenames, key=lambda filename: int(os.path.splitext(filename)[0]))]
            
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.img_filename[index]

    def __len__(self):
        return len(self.img_filename)

class FeatExtractor(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(FeatExtractor, self).__init__()
        if model_name == 'vgg11':
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            cl1.weight = original_model.classifier[0].weight
            cl1.bias = original_model.classifier[0].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[3].weight
            cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
            )
            self.model_name = 'vgg11'
        if model_name == "vgg16":
            original_model = models.vgg16_bn(pretrained)
            self.features = original_model.features
            cl_0 = nn.Linear(25088, 4096)
            cl_1 = nn.Linear(4096, 4096)
            if pretrained:
                cl_0.weight = original_model.classifier[0].weight
                cl_0.bias = original_model.classifier[0].bias
                cl_1.weight = original_model.classifier[3].weight
                cl_1.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl_0,
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                cl_1,
                nn.ReLU(inplace=True),
            )
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.classifier.parameters():
                p.requires_grad = False
            self.model_name = 'vgg16'
            
        if model_name == "resnet50":
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            for p in self.features.parameters():
                p.requires_grad = False
            self.model_name = 'resnet50'


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
            f = self.classifier(f)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
            f = self.classifier(f)
        if self.model_name == "resnet50":
            f = f.view(f.size(0), -1)
        return f

if __name__=="__main__":
    DATA_DIR = './retrieval_test'
    model = FeatExtractor('resnet50')
    model.cuda()
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_loader = torch.utils.data.DataLoader(
             MyDataset(DATA_DIR, transforms.Compose([
             transforms.Resize((224,224)),
             transforms.ToTensor(),
             normalize,
         ])),
         batch_size=300,
         shuffle=False,
         num_workers=4,)
    all_feat = []
    all_name = []
    for i, (img, img_name) in enumerate(train_loader):
        img_var = Variable(img.cuda())
        print('Block '+str(i)+' running...')
        output = model(img_var)
        all_feat.extend(output.cpu().data.numpy())
        all_name.extend(list(img_name))
    np.save('all_feat_res50', all_feat)
    np.save('all_name_res50', all_name)