#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:47:31 2023

@author: dennischang
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
episode = 100
window_size = 20

def move_average(data):
    return np.convolve(data, np.ones(int(window_size)) / float(window_size), 'same')

def draw(data):
    # sns.set(style="darkgrid", font_scale=1.5)
    fig = plt.figure()
    datas = np.load(f'train_record_gua/{data}_{episode}.npy', allow_pickle=True)
    y = move_average(datas[:500])
    
    t = [[index,value] for (index,value) in enumerate(y)] 
    df = pd.DataFrame(t,columns=['steps','loss'])
    # x = np.arange(len(datas))
    sns.lineplot(data=df,x='steps',y='loss',err_style='band')
    
if __name__ == '__main__':
    draw('loss')