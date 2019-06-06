#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from cnn_model import FeatExtractor
import numpy as np
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
from torchvision import models
import torch.nn as nn
from scipy.spatial import distance
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

all_feat = np.load('all_feat_res50.npy')
all_name = np.load('all_name_res50.npy')
model = FeatExtractor('resnet50')

model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
basic_trans = transforms.Compose([
             transforms.Resize((224,224)),
             transforms.ToTensor(),
             normalize,
         ])

def extractOnce(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = basic_trans(img).unsqueeze(0)
    output = model(Variable(img))
    return output.data.numpy()

def returnRes(img_path, top_k, fname, vis):
    query = extractOnce(img_path)
    dis = distance.cdist(query, all_feat, 'cosine')
    rank = np.argsort(dis)[0:]
    img_name_rank = all_name[rank]
    similarity_value = 1-dis
    if vis:
        imglist = []
        for f in img_name_rank[:top_k]:
            img = Image.open(os.path.join('./retrieval_test',f))
            img = img.convert('RGB')
            img = transforms.Compose([
                 transforms.Resize((224,224)),
                 transforms.ToTensor(),
             ])(img)
            imglist.append(img)
        #save_image(imglist, filename=fname,nrow=10)
    img_rank = rank[0,0:top_k]
    img_rank = img_rank.reshape((1,top_k))
    img_rank = img_rank[np.newaxis,:]
    return img_rank


def returnRank(root_dir, top_k, vis):
    n = len([name for name in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, name))])
    img_rank = np.zeros((n, top_k))
    file_list = os.listdir(root_dir)
    i = 0
    for f in file_list:
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            img_list = returnRes(file_path, top_k, 'test.jpg', vis)
            img_rank[i,:] = img_list
            i = i+1
    return img_rank





if __name__ == "__main__":
    top_k = 50
    vis = False
    rotation_dir = 'D:\\EESM6980project\\retrival-test\\test\\znew_try\\r_20_new'
    
    img_list = returnRank(rotation_dir, top_k, vis)
    
    np.save('r20_new.npy',img_list)



    '''filename=('./baseline_rank.txt')
    file = open(filename, 'w')
    for i in range(len(img_list)):
        file.write(str(img_list[i])+'\n')
    file.close()'''
 
    '''img_base = '51.jpg'
    #img_query = '4.jpg'
    rank_base = returnRes(img_base, top_k,'Digital.jpg', vis)
    print(rank_base.shape)'''
    #(rank_query,similarity_value2) = returnRes(img_query, top_k, 'bb_221.jpg',vis)
    #r_topk = np.where((np.isin(rank_query, rank_base)) == True, 1, 0)