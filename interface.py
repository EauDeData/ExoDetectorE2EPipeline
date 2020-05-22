#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:16:52 2020

@author: gerard
"""


import tkinter as Tk
import skyCleaner, skyAlignment, skyTracker, skyBrian
from PltInterface import Plot
import matplotlib.pyplot as plt
import numpy as np

class CreationMenu:
    def __init__(self):
        self.master = Tk.Tk()
        self.master.title("EDE2EP - ExoDetectorE2EPipeline")
        self.master.geometry("700x400")
        self.master.resizable(width=False, height=False)

        self.observation = Tk.Label(self.master, text="Folder containing the observation:", width=50)
        self.observation.pack()

        self._input_observation = Tk.Entry(self.master)
        self._input_observation.pack()

        self.df = Tk.Label(self.master, text="Folder containing the dark frames:", width=50) #flat dark 
        self.df.pack()

        self._input_df = Tk.Entry(self.master)
        self._input_df.pack()

        self.ff = Tk.Label(self.master, text="Folder containing the flat frames:", width=50)
        self.ff.pack()

        self._input_ff = Tk.Entry(self.master)
        self._input_ff.pack()

        self.fdf = Tk.Label(self.master, text="Folder containing the flat dark frames:", width=50)
        self.fdf.pack()

        self._input_fdf = Tk.Entry(self.master)
        self._input_fdf.pack()
        
        self._load = Tk.Button(self.master, command = self.load_button, text = "Load images", bg = 'Orchid1')
        self._load.pack()
        self._load.place(x=300, y=140)

        self._calculate_lc = Tk.Button(self.master, command = self.calculate_button, text = "Get light curve from star", bg = 'Orchid2')
        self._calculate_lc.pack()
        self._calculate_lc.place(x=260, y=175)
        
        self._detect_exo = Tk.Button(self.master, command = self.detect_button, text = "Detect exoplanet", bg = 'Orchid3')
        self._detect_exo.pack()
        self._detect_exo.place(x=285, y=215)

        self._canvas = Tk.Canvas(self.master, height=100, width=600, bg="snow", highlightbackground="gainsboro")
        self.set_canvas_text()
        self._canvas.place(x=50, y=255)
        
        self.quit = Tk.Button(master=self.master, text="Quit", command=self._quit)
        self.quit.pack(side=Tk.BOTTOM)

        self.master.mainloop()

    def set_canvas_text(self, text = '\n\tBenvingudes!'
                        '\n\n\tIntroduiu les dades per obtenir un resultat!'):

        self._canvas.delete('Out')
        self._canvas.create_text(220,30,tag = 'Out', text = text)

        return None
    
    def load_button(self):
        obs = str(self._input_observation.get())
        df = str(self._input_df.get())
        ff = str(self._input_ff.get())
        fdf = str(self._input_fdf.get())

        self.images, df, ff, fdf = skyCleaner.loadImages(obs, df, ff, fdf)
        #cleanImages = skyCleaner.cleanImages(images, df, ff, fdf)
        cleanImages = self.images
        self.alignedImages = skyAlignment.cross_correlation_fourier(cleanImages)
        self.starCoords , self.viewCoords = skyTracker.getStarCoords(self.images[0])
        print("Getting curves...")
        lightCurve = skyBrian.getBrians(self.images, self.starCoords)
        self.lightCurves = skyBrian.getBetterBrians(lightCurve)
        print("Done!")
        

    def calculate_button(self):
        #Calcularà la corva de llum de les estrelles seleccionades

        images = self.alignedImages
        betterLightCurve = self.lightCurves
        plt.title("Select the stars you want to get the light curve from:")
        plt.figure(1,figsize=(10,10))
        #plt.scatter(self.viewCoords[0], self.viewCoords[1], s=80, facecolors='none', edgecolors='r')
        plt.imshow(images[0], cmap="gray")
        coords = plt.ginput(-1)
        plt.show()
        plt.close(1)
        coords = np.array(coords)
        #coords[:,[0, 1]] = coords[:,[1, 0]]
        starCoords = np.array(self.starCoords)
        selectedCurves = []
        selectedStars = []
        for star in coords:
            dist = np.sqrt(np.sum(np.power((starCoords - star), 2),1))
            idx = np.argmin(dist)
            selectedCurves.append(betterLightCurve[idx])
            selectedStars.append(starCoords[idx])
        print("N stars:", len(selectedCurves))
        Plot(selectedCurves, selectedStars)
        
        return None
            
    def detect_button(self):
        #Detectara la possible estrella amb un exoplaneta

        """
        exo = CEP.Exo_nou()
        plotter = GG.Plotter(ETS.Get_exo_list())
        try:
            exo.set_values(float(self._input_mass.get()), float(self._input_radius.get()), float(self._input_temperature.get()))
            mediocrity = exo.get_mediocrity()
            esi = exo.getESI()

            self.set_canvas_text(str("L'exoplaneta creat té una mediocritat del " + str(mediocrity) + '%') +
                                 str("\nEl mateix té una semblança a la tera del " + str(esi) + '%')+
            '\nA més, segons el model KMeans, aquest pertany a la clase etiquetada com '+str(plotter.K_Means(exo)))
        except ValueError:
            self.set_canvas_text(text="S'han d'introduïr dades vàlides")
        """

        return None
    
    def _quit(self):
        self.master.quit()     # stops mainloop
        self.master.destroy()
    
adri = CreationMenu()