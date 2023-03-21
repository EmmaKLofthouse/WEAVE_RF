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
import plots as pl
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

    fluxdata = []
    Ns_CIV_data, zs_CIV_data = [], []
    Ns_MgII_data, zs_MgII_data  = [], []

    # Loop over hdf5 files
    for hdf5file in specfiles:

        print("Working with file: %s"%hdf5file)
        
        # Load file for nsight sightlines
        data = h5py.File(hdf5file)

        #QSO info
        zems = np.array(data['z_qso']) #Of shape nsight
        Rqsos = np.array(data['R_qso']) #Of shape nsight

        #Spectra
        wave = np.array(data['wave'])
        fluxes = np.array(data['flux']) #Of shape nsight x npix
            
        #Column densities
        Ncat = data['absorbers'] #Group of nsight numpy recarrays

        # Loop over each sightline
        for ind in range(len(Ncat)):
            
            sight = '%s'%ind
            
            # Get qso properties
            zqso = zems[ind]
            Rqso = Rqsos[ind]
            
            # Get flux
            flux = fluxes[ind, :]

            zMgII, NMgII, zCIV, NCIV, zMgII_clean, NMgII_clean, zCIV_clean, NCIV_clean =  extract_abs_properties(ind, sight, Ncat, wave)
            
            # Process through WEAVEify
            out_wave, out_flux, out_error, in_wave, in_flux = add_noise(wave, flux, Rqso, zqso)
            #out_wave, out_flux = wave, flux
            #plot_check(zMgII, zMgII_clean, zCIV, zCIV_clean, NMgII, NCIV, out_wave, out_flux,  out_error, in_wave, in_flux)
           
            # Drop systems where flux is nan
            if not all(np.isfinite(out_flux)):
                continue
                
            fluxdata.append(list(out_flux))
            Ns_CIV_data.append(list(NCIV))
            zs_CIV_data.append(list(zCIV))
            Ns_MgII_data.append(list(NMgII))
            zs_MgII_data.append(list(zMgII))

    #convert to velocity-space, relative to wave[0] which is 3700A
    vel = [0]
    for w in range(1,len(out_wave)):
        wavestep = (out_wave[w]-out_wave[w-1])/out_wave[w]
        velstep = wavestep *_c
        vel.append(vel[w-1] + velstep)
    
    specDict = dict(Flux  = fluxdata,
                    NCIV  = Ns_CIV_data,
                    zCIV  = zs_CIV_data,
                    NMgII = Ns_MgII_data,
                    zMgII = zs_MgII_data,
                    wave  = list(out_wave),
                    vel   = list(vel))

    return specDict

def extract_abs_properties(ind, sight, Ncat, wave):
    """ 
    For each sightline in the hdf5 files, extract the needed information
    """
    
    #Load recarray for sightline

    #WARNING -- hdf5 files have duplicate information in Nsight about absorbers.
    #This issues was due to an extra indentation while populating the Ncat[sight] table
    #leading to each previous absorber being re-added. This DOES NOT add additional
    #absorption lines in the spectra though (phew), ***so just need to use np.unique**
    #In order to remove duplicate information... Will fix once mocks are completed.

    Nsight = np.unique(np.array(Ncat[sight])) #Remove duplication here...
    zabs = Nsight['zabs']
    ions = Nsight['ion']
    logNs = Nsight['logN']

    #print(Nsight)

    #Get MgII columns and redshifts
    inds = np.argwhere(ions == b'MgII')[:,0]
    zMgII = zabs[inds]
    NMgII = logNs[inds]

    #Get CIV columns and redshifts
    inds = np.argwhere(ions == b'CIV')[:,0]
    zCIV = zabs[inds]
    NCIV = logNs[inds]

    ################################################################################
    #WARNING -- a column density/redshift might be outputted but it's not covered by
    #the wavelength range of the spectrum! May want to clean the absorber catalogue a bit...

    #Wavelength of the doublets
    MgII_lams = [2796.3543, 2803.5315]
    CIV_lams = [1548.2040, 1550.7810]

    #Ensure MgII doublet is within wavelength range of spectrum
    MgII_cut_inds = (MgII_lams[0]*(1.0+zMgII) > np.min(wave)) * (MgII_lams[1]*(1.0+zMgII) < np.max(wave))
    #Clean the absorber catalogue
    zMgII_clean = zMgII[MgII_cut_inds]
    NMgII_clean = NMgII[MgII_cut_inds]
    #Do same for CIV doublet
    CIV_cut_inds = (CIV_lams[0]*(1.0+zCIV) > np.min(wave)) * (CIV_lams[1]*(1.0+zCIV) < np.max(wave))
    zCIV_clean = zCIV[CIV_cut_inds]
    NCIV_clean = NCIV[CIV_cut_inds]
    
    return zMgII, NMgII, zCIV, NCIV, zMgII_clean, NMgII_clean, zCIV_clean, NCIV_clean

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

def plot_check(zMgII, zMgII_clean, zCIV, zCIV_clean, NMgII, NCIV, out_wave, out_flux, out_error, in_wave, in_flux):
    """
    analysis code to plot and make sure it works
    """
    
    import matplotlib.pyplot as plt

    for ii, z in enumerate(zMgII):
        if z in zMgII_clean:
           print("MgII at z=%.5f (logN=%.2f) -- in spectrum"%(z, NMgII[ii]))
        else:
           print("MgII at z=%.5f (logN=%.2f) -- outside spectrum"%(z, NMgII[ii]))
    for ii, z in enumerate(zCIV):
        if z in zCIV_clean:
           print("CIV at z=%.5f (logN=%.2f) -- in spectrum"%(z, NCIV[ii]))
        else:
           print("CIV at z=%.5f (logN=%.2f) -- outside spectrum"%(z, NCIV[ii]))
    plt.figure()

    plt.plot(out_wave, out_flux, 'k')
    plt.plot(out_wave, out_error, 'r', alpha=0.5)
    plt.plot(in_wave, in_flux, 'c')

    plt.show()
    plt.close()
    
    return

########################################################
print("Reading spectra and adding noise...")

training_list = '/data/tberg/WEAVE/Working/training_mocks.lst'
#training_list = 'training_mocks_short10.lst'
analysis_list = '/data/tberg/WEAVE/Working/analysis_mocks.lst'

#fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data = read_spectra(training_list, analysis_list)
trainSpec = read_spectra(training_list)
#analysisSpec = read_spectra(analysis_list)

# Save spectra with noise
import json
with open("trainSpec.json", "w") as write_file:
    json.dump(trainSpec, write_file, indent=4)
#with open("analysisSpec.json", "w") as write_file:
#    json.dump(analysisSpec, write_file, indent=4)



