import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.decomposition import NMF 
from multiprocessing import Pool
from pathlib import Path
import os
import sys
import scipy.constants as const
import pandas as pd
import h5py
import glob
#Add homebrew routines to pythonpath
#Temporarily add a "package" called WEAVE to pythonpath
import sys
sys.path.append('/home/tberg/Py3/')
#Import weavify from said WEAVE package
from WEAVE import weavify
import utils as utl

_c = const.c/1000

np.seterr(invalid='ignore')

def read_spectra(filelist):

    # Get all files to be used for training
    specfiles = []
    with open(filelist, 'r') as f:
        for line in f:
            specfiles.append(line.strip())

    specfiles.sort()

    # Loop over hdf5 files
    for hdf5file in specfiles:

        print("Working with file: %s"%hdf5file)
        
        # Load file for nsight sightlines
        data = h5py.File("/data/tberg/WEAVE/Working/" + hdf5file)
        #data = h5py.File(hdf5file)
        #QSO info
        zems = np.array(data['z_qso']) #Of shape nsight
        Rqsos = np.array(data['R_qso']) #Of shape nsight

        #Spectra
        wave = np.array(data['wave'])
        fluxes = np.array(data['flux']) #Of shape nsight x npix
            
        #Column densities
        Ncat = data['absorbers'] #Group of nsight numpy recarrays

        fluxdata, wavedata, errdata = [], [], []

        # Loop over each sightline
        for ind in range(len(Ncat)):
            
            sight = '%s'%ind
            
            # Get qso properties
            zqso = zems[ind]
            Rqso = Rqsos[ind]
            
            # Get flux
            flux = fluxes[ind, :]

            # Process through WEAVEify
            out_wave, out_flux, out_error, in_wave, in_flux = add_noise(wave, flux, Rqso, zqso)
           
            # Set systems where flux is nan to placeholder 
            if all(np.isfinite(out_flux)):
                fluxdata.append(out_flux)
                wavedata.append(out_wave)
                errdata.append(out_error)
            else:
                fluxdata.append(np.array([-999.]*23671)) 
                wavedata.append(np.array([-999.]*23671))
                errdata.append(np.array([-999.]*23671))      

        # Save as hdf5
        filename = hdf5file.split('NMFPM_data/')[1]
        index = filename.find('.')
        new_file = filename[:index] + '_weavified' + filename[index:]
        new = h5py.File('NMFPM_data/' + new_file,'w')

        for key in data.keys():
            data.copy(key, new)
        new.create_dataset('flux_weavify', data=fluxdata)
        new.create_dataset('wave_weavify', data=wavedata)
        new.create_dataset('error_weavify', data=errdata)
        new.close()


    #return fluxdata, wavedata, data #specDict

def add_noise(wave, flux, Rqso, zqso):
    """
    Use WEAVIFY to add realistic noise to each spectrum
    
    """
    #############PADDING###############
    #WEAVEify wavelength range doesn't always agree with the spectra generated from Trystyn's mocks
    #Padding the input spectrum by +/- 500A fixes the issue
    #Get the min/max wavelength
    inmin = np.min(wave)
    inmax = np.max(wave)
    #Create two arrays to pad the lower and higher wavelength range by 500A
    wpad_lo = np.arange(inmin-500,inmin,1.0)
    wpad_hi = np.arange(inmax,inmax+500,1.0)
    #Pad the wavelength array with the two arrays
    in_wave = np.concatenate((wpad_lo, wave, wpad_hi))
    #Pad the flux array to include a flux of 1 in the added wavelength range
    in_flux = np.concatenate((np.zeros(len(wpad_lo))+1.0, flux, np.zeros(len(wpad_hi))+1.0))

    #####Run WEAVEify on the spectrum#######
    out_wave, out_flux, out_error = weavify.weavify_spectrum(in_wave, in_flux, normalize=True, m_source=Rqso, z_em=zqso, wave_band_source='R')

    return out_wave, out_flux, out_error, in_wave, in_flux


########################################################
print("Reading spectra and adding noise...")

training_list = '/data/tberg/WEAVE/Working/training_mocks.lst'
#training_list = 'training_mocks_short.lst'
analysis_list = '/data/tberg/WEAVE/Working/analysis_mocks.lst'

#fluxdata, wavedata, alldata = read_spectra(training_list)
read_spectra(training_list)
#analysisSpec = read_spectra(analysis_list)




#Save as json
#with open("analysisSpec.json", "w") as write_file:
#    json.dump(analysisSpec, write_file, indent=4)



