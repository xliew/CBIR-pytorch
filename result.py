# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:57:53 2019

@author: 李霄雯
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score



def Analysis(y_true,y_pred):
    r = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return r,f1

def AP(y_true, y_pred):
    precision = []
    p = []
    r = []
    for i in range(686):
        x = 0
        y = 0
        b = 0
        r1 = 0
        for z in range(8):
            for j in range(8):
                if y_pred[i,z]==y_true[i,j]:
                    y = (x+1)/(z+1)+y
                    b = (x+1)/(z+1)
                    x = x+1
                    r1 = (x+1)/8
                    r.append(r1)
                    p.append(b)
                
        if x!=0:
            ap = y/x
        else:
            ap = 0
        precision.append(ap)
    return(precision)

def Pr(y_true, y_pred):
    precision = []
    recall = []
    for i in range(10):
        x = 0
        b = 0
        r1 = 0
        for z in range(50):
            for j in range(50):
                if y_pred[i,z]==y_true[i+676,j]:
                    b = (x+1)/(z+1)
                    x = x+1
                    r1 = (x+1)/50
                    recall.append(r1)
                    precision.append(b)
    return(precision, recall)





'''def Recall(y_true, y_pred):
    for i in range(686):
        x = 0
        rx = 0
        for j in range(7):
            if y_true[i,j]==y_pred[i,j]:
                p1 = (x+1)/(j+1)
                px = px+p1
                x = x+1
        ap = px/(x+1)
        recall.append(ap)
    return(recall)'''
    



if __name__ == "__main__":
    recall = []
    f1 = []
    blur_rank = np.load("blur_rank.npy")
    illu_rank = np.load("illu.npy")
    rotation_20 = np.load("rotation_20.npy")
    baseline_new = np.load("baseline_new.npy")
    s_50_1 = np.load("s_50_1.npy")
    s_15_1 = np.load("s_15_1.npy")
    ro_40 = np.load("r_40.npy")
    p_1_1 = np.load("p_1_1.npy")
    p_2_1 = np.load("p_2_1.npy")
    p_3_1 = np.load("p_3_1.npy")
    
    
    baseline_new = np.load("baseline_new.npy")
    illu = np.load("illu_2.npy")
    
    i_2 = AP(baseline_new, illu)
    i_2_p = sum(i_2)/len(i_2)
    
    
    p_1_100 = np.load("p_1_100.npy")
    baseline_50 = np.load("baseline_50.npy")
    blur_new = np.load("blur_new.npy")
    illu_new = np.load("illu_new.npy")
    p_2_1_new = np.load("p_2_1_new.npy")
    p_3_1_new = np.load("p_3_1_new.npy")
    r_20_new = np.load("r_20_new.npy")
    r_40_new = np.load("r_40_new.npy")
    r_60_new = np.load("r_60_new.npy")
    s_15_new = np.load("s_15_new.npy")
    s_50_new = np.load("s_50_new.npy")
    r20_new = np.load("r20_new.npy")
    pre,rec = Pr(baseline_50, p_1_100)
    #pre_p2,rec_p2 =Pr(baseline_50, p_2_1_new)
    #pre_p3,rec_p3 = Pr(baseline_50,p_3_1_new)
    #pre_b,rec_b =Pr(baseline_50,blur_new)
    #pre_i,rec_i =Pr(baseline_50,illu_new)
    #pre_s50,rec_s50 =Pr(baseline_50,s_50_new)
    #pre_s15,rec_s15 =Pr(baseline_50,s_15_new)
    pre_r20,rec_r20 =Pr(baseline_50,r20_new)
    #pre_r40,rec_r40 =Pr(baseline_50,r_40_new)
   # pre_r60,rec_r60 =Pr(baseline_50,r_60_new)
    
    '''for i in range(686):
        (r,f) = Analysis(baseline_new[i],s_50_1[i])
        recall.append(r)
        f1.append(f)'''
   # precision, s_p, s_r = AP(baseline_new,s_50_1)
    
    '''a = 0
    x = 0
    for i in range(len(precision)):
        if precision[i]>0.2 :
            x = x+precision[i]
            a = a+1
    pre = x/a'''
    p_1 = AP(baseline_new, p_1_1)
    p_1_p = sum(p_1)/len(p_1)
    
    p_2 = AP(baseline_new, p_2_1)
    p_2_p = sum(p_2)/len(p_2)
    p_3 = AP(baseline_new, p_3_1)
    p_3_p = sum(p_3)/len(p_3)
    
    #r_20 = AP(baseline_new, rotation_20)
    #r_40 = AP(baseline_new, ro_40)
    # r_p = sum(r_40)/len(r_40)
    precision_15_1 = AP(baseline_new,s_15_1)
    blur_precision = AP(baseline_new, blur_rank)
    #illu = AP(baseline_new, illu_rank)
    
    
    blur_map = sum(blur_precision)/len(blur_precision)
    #pre_50_1 = sum(precision)/len(precision)
    #pre_15_1 = sum(precision_15_1)/len(precision_15_1)
    '''for i in range(len(precision_15_1)):
        if precision_15_1[i]>0 :
            x = x+precision_15_1[i]
            a = a+1
    pre_15 = x/a'''
    '''for i in range(len(illu)):
        if illu[i]>0 :
            x = x+illu[i]
            a = a+1
    illu_new = x/a'''
    '''for i in range(len(r_40)):
        if r_40[i]>0 :
            x = x+r_40[i]
            a = a+1
    r_new = x/a'''
    