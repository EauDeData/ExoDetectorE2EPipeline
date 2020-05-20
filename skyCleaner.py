#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:14:52 2020

@author: gerard
"""
""" NOTES
- A prtir de la 100 inclosa estan netes, abans brutes jasjas
"""
#### Image processing ####
import cv2
import matplotlib.pyplot as plt
from PIL import Image as pil
import seaborn as sns


#### Data sctructure and numerical support #####
import numpy as np
from scipy import signal as ss

#### OS Support ####
import os

### Astro data support ###
from astropy.io import fits

#### Data Flags for classification and usage #####

FLAT_FRAMES_EXPTIME = '5s'
FOLDER = './cleanObservations/' #Where the observation is placed (wasp52b) (cleanObservations)
IMAGENAME_FLAG = '' #'TRE' #How we will identify images on a folder ('')
FDO_FLAG = 'CDO' #Flat Dark frames identifier
FFF_FLAG = 'FFF' #Flat Frames identifier
DF_FLAG = 'CDO' #Dark Frames identifier

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

def histogram(raw_image, bins = 800, hist = True, kde = True): #hist: normed hist; kde = gaussian kernel density estimate
    # Plot a histogram of the distribution of the pixels
    sns.distplot(raw_image.ravel(), bins = bins, hist = hist,
                 label=f'Pixel Mean {np.mean(raw_image):.4f} & Standard Deviation {np.std(raw_image):.4f}', kde = kde)
    plt.legend(loc='upper center')
    plt.title('Distribution of Pixel Intensities in the Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# Pixels in Image')
               
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


##### Image batching ####

allFiles = os.listdir(FOLDER) #All docs names in the folder
images = list(filter(lambda x: IMAGENAME_FLAG in x, allFiles)) #List of the images
images.sort() #Ordenem
fdf = list(filter(lambda x: FDO_FLAG in x and FLAT_FRAMES_EXPTIME in x, allFiles)) #List of flat dark frames
ff = list(filter(lambda x: FFF_FLAG in x, allFiles)) #List of flat frames
df = list(filter(lambda x: DF_FLAG in x and not FLAT_FRAMES_EXPTIME in x, allFiles)) #List of dark frames

print(" ---- Data Summary -----")
print("We have identified {} files as {}".format(len(images), 'images'))
print("We have identified {} files as {}".format(len(fdf), 'flat dark frames'))
print("We have identified {} files as {}".format(len(ff), 'flat frames'))
print("We have identified {} files as {}".format(len(df), 'dark frames'))

images = [FOLDER + name for name in images]
fdf = [FOLDER + name for name in fdf]
ff = [FOLDER + name for name in ff]
df = [FOLDER + name for name in df]

images = [fits.open(image)[0].data for image in images] #les imatges
fdf = [fits.open(image)[0].data for image in fdf] #Flat dark frames son els dark frams de l'exposició dels flat frames
ff = [fits.open(image)[0].data for image in ff] #Flat frames son frames totalment iluminats per trobar pixels morts (trobarem pols a la ccd)
df = [fits.open(image)[0].data for image in df] #Dark frames són les imatges apuntant a la foscar (trobar el soroll de fons)

#### Test zone ####
"""
imshow(images[10], lut = 'Sí, si us plau') #Amb laLUT passada per que es vegi guai l'imatge
imshow(fdf[10])
imshow(ff[10], lut = True)
imshow(df[10], lut = True)
"""

#TODO: Neteja del senyal
# NETEJAR DF
def cleanDF(images, DF):
    nframes = len(DF)
    meanIm = sum(DF)/nframes
    # Convert to range 0-255
    meanIm = (meanIm - meanIm.min())
    meanIm = 255 * meanIm/meanIm.max()
    # Double threshold to get dark and bright peaks
    light =  (40 > meanIm) + (meanIm > 210) #TODO: mirar valors
    # Remove noise
    cleanImages = [images[i] - meanIm * light for i in range(len(images))]
    #imshow(res.astype(int), lut = True)
    #imshow(images[10] != res)
    return cleanImages

def cleanImages(images, df, ff, fdf):
    newImages = cleanDF(images, df)
    #images = cleanFF(images, ff, fdf)
    return newImages

""" Intento fallido
def cleanFDF(images, FDF):
    nframes = len(FDF)
    meanIm = sum(FDF)/nframes
    # Convert to range 0-255
    meanIm = (meanIm - meanIm.min())
    meanIm = 255 * meanIm/meanIm.max()
    # Double threshold to get dark and bright peaks
    light =  (40 > meanIm) + (meanIm > 210) #TODO: mirar valors
    # Remove noise
    cleanImages = [images[i] - meanIm * light for i in range(len(images))]
#    imshow(cleanImages[10].astype(int), lut = True, index = 0)
#    imshow(images[10] != cleanImages[10], index = 2)
    return cleanImages

#Netejem FF amb FDF

def cleanFF(images, FF, FDF):
    newFF = cleanFDF(FF, FDF)
    nframes = len(newFF)
    meanIm = sum(newFF)/nframes
    # Convert to range 0-255
    meanIm = (meanIm - meanIm.min())
    meanIm = 255 * meanIm/meanIm.max()
    # Double threshold to get dark and bright peaks
    light =  (meanIm > 100) * (meanIm < 100) #TODO: mirar valors
    # Remove noise
    cleanImages = [images[i] - meanIm * light for i in range(len(images))]
    #imshow(cleanImages[0].astype(int), lut = True, index = 0)
    #imshow(images[0].astype(int), lut = True, index = 1)
    return cleanImages
    
    
cleanImages = cleanFF(newImages, ff, fdf)
imshow(cleanImages[0].astype(int), lut = True)
"""

