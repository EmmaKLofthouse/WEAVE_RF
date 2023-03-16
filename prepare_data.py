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

def read_spectra(training_list, test_list):

    # Get all files to be used for training
    specfiles = []
    with open(training_list, 'r') as f:
        for line in f:
            specfiles.append(line.strip())

    specfiles.sort()

    fluxdata = []
    Ns_CIV_data, zs_CIV_data = [], []
    Ns_MgII_data, zs_MgII_data  = [], []

    # Loop over hdf5 files
    for hdf5file in specfiles[:2]:

        print("Working with file: %s"%hdf5file)
        
        # Load file for nsight sightlines
        data = h5py.File(hdf5file)

        #QSO info
        zems = np.array(data['z_qso']) #Of shape nsight
        Rqsos = np.array(data['R_qso']) #Of shape nsight

        #Spectra
        wave = np.array(data['wave'])
        fluxes = np.array(data['flux']) #Of shape nsight x npix

        #convert to velocity-space, relative to wave[0] which is 3700A
        vel = [0]
        for w in range(1,len(wave)):
            wavestep = (wave[w]-wave[w-1])/wave[w]
            velstep = wavestep *_c
            vel.append(vel[w-1] + velstep)
            
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
        
    return fluxdata, out_wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data 

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
    Test code to plot and make sure it works
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

def slice_input(data, wave, vel, slide_idx):
    
    fluxdata     = data['Flux']
    Ns_CIV_data  = data['NCIV']
    zs_CIV_data  = data['zCIV']
    Ns_MgII_data = data['NMgII']
    zs_MgII_data = data['zMgII']

    # Slice spectrum into small regions and add tag for whether there is/isnt an absorber
    fluxslices = []
    velslices = []
    waveslices = []

    is_abs = []    
    absInfo = []    # will contain logNs and redshifts of absorbers

    for source in range(len(fluxdata)):

        Ns_CIV = Ns_CIV_data[source]
        Ns_MgII = Ns_MgII_data[source]

        zs_CIV = zs_CIV_data[source]
        zs_MgII = zs_MgII_data[source]

        spec = fluxdata[source]

        # Determine observed wavelengths of absorbers
        obs_CIV_1548_wave = 1548*(zs_CIV + 1)
        obs_CIV_1550_wave = 1550*(zs_CIV + 1)

        obs_MgII_2796_wave = 2796.4*(zs_MgII + 1)
        obs_MgII_2803_wave = 2803.5*(zs_MgII + 1)
       
        startidx = 0 #initialise first chunk
        num_idxs = 100 #int(2000./velstep)  #size of window

        while startidx + num_idxs < len(spec):
            
            flux_slice = spec[startidx:startidx+num_idxs] 
            vel_slice = vel[startidx:startidx+num_idxs]
            wave_slice = wave[startidx:startidx+num_idxs]
            
            startidx += slide_idx # number of indices to shift window by

            # Record if there is an absorber or not
            CIV1548_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_CIV_1548_wave]
            CIV1550_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_CIV_1550_wave]    

            MgII2796_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_MgII_2796_wave]    
            MgII2803_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_MgII_2803_wave]  

            # Add slice to array of inputs
            fluxslices.append(list(flux_slice))
            velslices.append(list(vel_slice))
            waveslices.append(list(wave_slice))
            
            # If there is no absorption present, flag it as 0
            allLines_present = CIV1548_present + CIV1550_present + MgII2796_present + MgII2803_present
            if True not in allLines_present:
                is_abs.append(0)
                absInfo.append(['-',0, 0, "spec_" + str(source)]) 
                continue

            # Check if there are multiple of the same line within the window
            if (sum(CIV1548_present) > 1) |(sum(CIV1550_present) > 1) | (sum(MgII2796_present) > 1) |(sum(MgII2803_present) > 1):
                is_abs.append(5)
                absInfo.append(['multiple of same',0, 0, "spec_" + str(source)]) 
                continue
                
            elif (True in CIV1548_present + CIV1550_present) & (True in MgII2796_present + MgII2803_present):
                is_abs.append(4)
                absInfo.append(['MgII+CIV',0, 0, "spec_" + str(source)]) 
                continue

            elif ((True in CIV1548_present) | (True in CIV1550_present)): 
                matchidx = np.where(CIV1548_present)[0] 
                if len(matchidx) > 0:
                    if CIV1550_present[matchidx[0]]:
                        is_abs.append(1)
                        absInfo.append(['CIV',Ns_CIV[matchidx[0]],zs_CIV[matchidx[0]], "spec_" + str(source)])
                        continue
                is_abs.append(3)
                absInfo.append(['partial CIV',0,0, "spec_" + str(source)])
                continue

            elif ((True in MgII2796_present) | (True in MgII2803_present)): 
                matchidx = np.where(MgII2796_present)[0]    
                if len(matchidx) > 0:
                    if MgII2803_present[matchidx[0]]:
                        is_abs.append(2)
                        absInfo.append(['MgII',Ns_MgII[matchidx[0]],zs_MgII[matchidx[0]], "spec_" + str(source)])
                        continue
                is_abs.append(3)
                absInfo.append(['partial MgII',0,0, "spec_" + str(source)])
            

    chunks = dict(fluxslices = fluxslices,
                  waveslices = waveslices, 
                  velslices = velslices, 
                  is_abs = is_abs, 
                  absInfo = absInfo)

    return chunks

def split_samples(fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data):

    # Change to train-test split from sklearn
    idx_split = int(len(fluxdata)*0.7)

    trainSpec = dict(Flux  = fluxdata[:idx_split],
                     NCIV  = Ns_CIV_data[:idx_split],
                     zCIV  = zs_CIV_data[:idx_split],
                     NMgII = Ns_MgII_data[:idx_split],
                     zMgII = zs_MgII_data[:idx_split])

    testSpec = dict(Flux  = fluxdata[idx_split:],
                    NCIV  = Ns_CIV_data[idx_split:],
                    zCIV  = zs_CIV_data[idx_split:],
                    NMgII = Ns_MgII_data[idx_split:],
                    zMgII = zs_MgII_data[idx_split:])


    return trainSpec, testSpec
    

########################################################
print("Reading spectra and adding noise...")

training_list = '/data/tberg/WEAVE/Working/training_mocks.lst'
test_list = '/data/tberg/WEAVE/Working/analysis_mocks.lst'

fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data = read_spectra(training_list, test_list)

print("Split train-test samples...")
trainSpec, testSpec = split_samples(fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data)

# Save spectra with noise
import json
with open("trainSpec.json", "w") as write_file:
    json.dump(trainSpec, write_file, indent=4)
with open("testSpec.json", "w") as write_file:
    json.dump(testSpec, write_file, indent=4)

"""
print("Slicing spectra...")
# Split spectra into chunks and assign flag for absorbers
# Use a fine sliding for train sample but larger for test to avoid duplication
trainChunks = slice_input(trainSpec, wave, vel, 5) # Give all the data and a 
                                                # value to shift the window by
testChunks = slice_input(testSpec, wave, vel, 50)
testFineChunks = slice_input(testSpec, wave, vel, 5)    

import json
with open("trainChunks.json", "w") as write_file:
    json.dump(trainChunks, write_file, indent=4)
with open("testFineChunks.json", "w") as write_file:
    json.dump(testFineChunks, write_file, indent=4)
with open("testChunks.json", "w") as write_file:
    json.dump(testChunks, write_file, indent=4)
"""
"""
# Save chunks to read into Random Forest Classifier in run_RF.py
df_train = pd.DataFrame(data=trainChunks)
df_train.to_pickle("trainChunks.pkl")
df_test = pd.DataFrame(data=testChunks)
df_test.to_pickle("testChunks.pkl")
df_testFine = pd.DataFrame(data=testFineChunks)
df_testFine.to_pickle("testFineChunks.pkl")
"""

