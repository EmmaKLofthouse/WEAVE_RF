import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier 
from pathlib import Path
import scipy.constants as const
import h5py
import glob
import pandas as pd
import joblib

np.seterr(invalid='ignore')

def rebin_spectrum(in_wave, in_flux, out_wave=None, density=True):
    """
    Rebin a spectrum with flux conservation (density=True) to the out_wave wavelength array.
    Code written by J. Trevor Mendle.

    Input
    -----
    in_wave (NumPy array)	
        The original wavelength array to be rebinned
    in_flux (NumPy array)	
        The original flux array to be rebined
    out_wave (None or Numpy array)	
        The rebinned wavelength array
    density (bool)			
        Whether to rebin treating flux as density (True) or not (False).
	    density=True (default) conserves flux density.
    
    Returns
    -------
    out_wave (NumPy array)
        The rebinned flux array.
    """

    if out_wave is None:
      print("Define a wavelength array to be returned")

    edges_in = (in_wave[:-1] + in_wave[1:]) / 2
    edges_in = np.r_[in_wave[0] - np.diff(in_wave)[0], np.r_[edges_in, in_wave[-1] + np.diff(in_wave)[-1]]]

    edges_out = (out_wave[:-1] + out_wave[1:]) / 2.
    edges_out = np.r_[out_wave[0] - np.diff(out_wave)[0], np.r_[edges_out, out_wave[-1] + np.diff(out_wave)[-1]]]

    if density:
      cumul = np.nancumsum(np.r_[0, np.diff(edges_in) * in_flux])
    else:
      cumul = np.nancumsum(np.r_[0, in_flux])
    cumul_rebinned = np.interp(edges_out, edges_in, cumul)

    if density:
      return np.diff(cumul_rebinned) / np.diff(edges_out)
    else:
      return np.diff(cumul_rebinned)


def read_hdf5_spec(filelist):

    # Get all files to be used for training
    specfiles = []
    with open(filelist, 'r') as f:
        for line in f:
            specfiles.append(line.strip())

    specfiles.sort()

    fluxdata = []
    sightline = []
    z_qso_data = []

    # Loop over hdf5 files
    for hdf5file in specfiles:

        print("Working with file: %s"%hdf5file)
        
        # Load file for nsight sightlines
        data = h5py.File(hdf5file, 'r')

        #QSO info
        zems = np.array(data['z_qso']) 

        #Spectra
        wave = np.array(data['wave_weavify'][0])
        fluxes = np.array(data['flux_weavify']) #Of shape nsight x npix

        #Column densities
        Ncat = data['absorbers'] #Group of nsight numpy recarrays

        # Loop over each sightline
        for ind in range(len(Ncat)):
            
            sight = '%s'%ind
            
            # Get qso properties
            zqso = zems[ind]
            # Get flux
            flux = fluxes[ind, :]
            

            # Drop systems where flux is all -999
            flux_check =  [flux[i] == -999. for i in range(len(flux))]
            if not all(flux_check):             
                fluxdata.append(list(flux))
                sightline.append(ind)
                z_qso_data.append(zqso)
    
    specDict = dict(Flux  = fluxdata,
                     wave  = list(wave),
                     orig_wave = data['wave'],
                     zqso  = z_qso_data,
                     sightline = sightline)
    
    return specDict


def slice_analysis_data(data, wave, slide_idx): 
    fluxdata       = data['Flux']
    zqso_data      = data['zqso']
    wave_constwave = data['wave']
    sightline_data = data['sightline']


    # Slice spectrum into small regions and add tag for whether there is/isnt an absorber
    fluxslices = []
    velslices = []
    waveslices = []
    sightlines = []

    for source in range(len(fluxdata)):
        if source%100 == 0:
            print("Slicing spectrum %s"%source + " out of %s"%len(fluxdata))

        # Rebin spectra onto constant velocity
        spec_constwave = fluxdata[source]
        spec = rebin_spectrum(np.array(wave_constwave), np.array(spec_constwave), 
                              out_wave=wave, density=True)

        zqso = zqso_data[source]
        wl_qso = 1215.67*(zqso + 1)        

        # Create slices of the spectra with associated flags
        startidx = 0 #initialise first chunk
        num_idxs = 100 #size of window

        while startidx + num_idxs < len(spec):
            
            wave_slice = wave[startidx:startidx+num_idxs]
            flux_slice = spec[startidx:startidx+num_idxs] 

            startidx += slide_idx # number of indices to shift window by

            # Exclude slice if it falls in the Lya forest
            if wave_slice[0] < wl_qso:
                continue

            # Add slice to array of inputs
            fluxslices.append(flux_slice)
            #velslices.append(vel_slice)
            waveslices.append(wave_slice)
            sightlines.append(sightline_data[source])

    return fluxslices, waveslices, sightlines

########################################################

analysis_list = "analysis_mocks.lst"
analysis_listSpec = read_hdf5_spec(analysis_list)

orig_wave_arr = np.array(analysis_listSpec['orig_wave'])

print("Slicing spectra...")
fluxslices, waveslices, sightlines = slice_analysis_data(analysis_listSpec,orig_wave_arr,25)

print("Loading the model...")
model = joblib.load("model_spec9557_EW0.2_withWeakFlag.joblib")

print("Making predictions...")
# Classify whether test sample are absorber or not
preds = model.predict(fluxslices)

# If you want confidence, return probability of classes
preds_probability = model.predict_proba(fluxslices)

# Create high confidence predictions
prob_cut = 0.3
preds_highConf = np.zeros(len(preds))

strongCIV = [((probs[1]>prob_cut) & (probs[1] == max(probs))) for probs in preds_probability]
strongMgII = [((probs[2]>prob_cut) & (probs[2] == max(probs))) for probs in preds_probability]
strongOther = [((max(probs[3:])>prob_cut) & (max(probs[3:]) == max(probs))) for probs in preds_probability]

preds_highConf[strongCIV] = 1
preds_highConf[strongMgII] = 2
preds_highConf[strongOther] = 3

print("Saving data...")
wave_cen = np.array(waveslices)[:,50]

d = {'wave_cen': wave_cen,  
     'preds':preds,
     'preds_probability': list(preds_probability),
     'preds_highConf': list(preds_highConf), 
     'sightline': list(sightlines)}

df = pd.DataFrame(data=d)
df.to_pickle("mock_samp650_predictions.pkl")

# Save only the predicted CIV and MgII
predsMask = (preds == 1) | (preds == 2)

d = {'wave_cen': np.array(wave_cen)[predsMask],  
     'preds':preds[predsMask],
     'preds_probability': list(preds_probability[predsMask]), 
     'sightline': list(sightlines[predsMask])}

df = pd.DataFrame(data=d)
df.to_pickle("mock_samp650_predicted_Abs_only.pkl")

# Save only the high confidence predicted CIV and MgII
predsMask = (preds_highConf == 1) | (preds_highConf == 2)

d = {'wave_cen': np.array(wave_cen)[predsMask],  
     'preds':preds[predsMask],
     'preds_highConf': list(preds_highConf[predsMask]),
     'preds_probability': list(preds_probability[predsMask]), 
     'sightline': list(sightlines[predsMask])}

df = pd.DataFrame(data=d)
df.to_pickle("mock_samp650_predicted_Abs_only_highConf" + str(prob_cut) + ".pkl")



