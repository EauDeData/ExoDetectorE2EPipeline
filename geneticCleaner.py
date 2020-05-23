#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:36:56 2020

@author: gerard
"""

import numpy as np
import random
import copy

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def evalue(image, ff, fdf, a, b, a2, b2, epsilon = 0.00000001):
    res = (ff / (fdf * abs(a2) + b2 + epsilon)) * abs(a) + b
    clean = image / (res + epsilon)
    return signaltonoise(clean.flatten())

def get_poblation(previous_bests):
    new_poblation = []
    for i in range(len(previous_bests)):
        newSample = copy.deepcopy(previous_bests[i])
        for j, value in enumerate(newSample):
            newSample[j] = (value - random.random()/10) + random.random()/10
        new_poblation.append(newSample)
    return new_poblation

def genesis():
    eva = [[1, 1, 1, 1]]
    for i in range(5):
        for new in get_poblation(eva):
            eva.append(new)
    return eva

def epoches(num, image, ff, fdf):
    poblation = genesis()
    loss = ['inf']*len(poblation)
    for i in range(num):
        for ng, gene in enumerate(poblation):
            loss[ng] =  evalue(image, ff, fdf, gene[0], gene[1], gene[2], gene[3])
        bests = sorted(poblation, key = lambda x: loss[poblation.index(x)])
        poblation = bests[:len(poblation)//2] + get_poblation(bests[:len(poblation)//2])
        print('EPOCH '+str(i)+' - Current best:', end=' ')
        for i in bests[0]:
            print(i, end = ' ')
    return bests