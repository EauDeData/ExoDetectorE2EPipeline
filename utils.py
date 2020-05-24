#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:38:17 2020

@author: gerard
"""

import numpy as np
import matplotlib.pyplot as plt

"""
This file contains all the functions used during the development of the project
"""

def normalize(image, bits = 8):
    tmp = image - image.min()
    tmp = tmp / tmp.max()
    tmp = tmp * ((2 ** bits) - 1)
    return tmp.astype(np.uint8)    

def imshow(im, c = 'gray', lut = False, log = False, index = 0, scatter = None):
    plt.figure(index)
    if scatter:
        plt.scatter(scatter[0], scatter[1])
    if lut:
        histim = loadHist(im)
        cumhist = cumHist(histim)
        im_luted = cumhist[im]
        plt.imshow(im_luted, cmap = c)
        plt.show()
        return True
    if log:
        plt.imshow(np.log(im + 1), cmap = c)
        plt.show()
    else:
        plt.imshow(im, cmap = c)
        plt.show()
    

def loadHist(im):
    maxim = im.max()
    minim = im.min()
    im = im-minim
    hist = np.zeros(maxim+1)
    vals = im.reshape(-1)
    for value in vals:
        hist[value] += 1
    return hist

def cumHist(hist):
    cummies = np.zeros(len(hist))
    cummies[0] = hist[0]
    for i in range(1, len(cummies)):
        cummies[i] = cummies[i-1] + hist[i]
    return cummies
              
def random_imshow(images):
    idx = np.random.randint(len(images), size=9)
    print("Idx:", idx)
    random_images = [images[x] for x in idx]
    plt.figure(figsize=(20,10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        histim = loadHist(random_images[i])
        cumhist = cumHist(histim)
        im_luted = cumhist[random_images[i]]
        plt.imshow(im_luted, cmap='gray')
        plt.axis('off')
        
def n_imshow(images, idx, n):
    print("Idx", idx, "to", n+idx)
    random_images = [images[x] for x in range(idx,n+idx,1)]
    plt.figure(figsize=(20,10))
    for i in range(n):
        plt.subplot((n//4)+1, 4, i + 1)
        histim = loadHist(random_images[i])
        cumhist = cumHist(histim)
        im_luted = cumhist[random_images[i]]
        plt.title(i+idx)
        plt.imshow(im_luted, cmap='gray')
        plt.axis('off')
        
def generate_gif(images, filename):
    import imageio
    frames = []
    for im in images:
        frames.append(normalize(im))
    imageio.mimsave('./' + str(filename) + '.gif', frames, duration=0.01)
    
def time_serie_plot(data, indx=0):
    plt.figure(indx)
    n_data = len(data)
    x = np.arange(n_data)
    plt.plot(x, data)
    
def imshow_all(images):
    img = None
    images = np.array(images).astype(int)
    p = 0
    for n in range(len(images)):
        im = images[n]
        print("Matrix")
        print(im)
        print("Mean:", np.mean(im))
        print("----")
        plt.title(p)
        if img is None:
            img = plt.imshow(im, cmap="gray")
        else:
            img.set_data(im)
        plt.draw()
        plt.pause(1)
        p+=1