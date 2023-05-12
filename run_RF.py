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
import json
from datetime import datetime

# Import from my files
import plots as pl
import utils as utl

_c = const.c/1000

np.seterr(invalid='ignore')

ions_dict = {'CII': [1334],
             'CIV': [1546, 1550],
             'OI': [1302],
             'MgII': [2796, 2803],
             'AlII': [1670],
             'SiII': [1260, 1304, 1526, 1808],
             'SiIV': [1393, 1402],
             'SII': [1250, 1253, 1259],
             'MnII': [2576, 2594],
             'FeII': [1608, 2600, 2382],
             'NiII': [1709, 1741],
             'ZnII': [2026],
             'HI': [1215.67,1025.72,972.53,949.74]
             }

def slice_input(data, wave, slide_idx, EW_threshold): #(data, wave, vel, slide_idx):
    
    fluxdata       = data['Flux']
    Ns_data        = data['Ndata']
    zs_data        = data['zdata']
    ions_data      = data['ionsdata']
    zqso_data      = data['zqso']
    wave_constwave = data['wave']
    EW_data        = data['EWdata']

    # Slice spectrum into small regions and add tag for whether there is/isnt an absorber
    fluxslices = []
    velslices = []
    waveslices = []

    is_abs = []    
    absInfo = [] # will contain logNs and redshifts of absorbers

    for source in range(len(fluxdata)):
        if source%100 == 0:
            print("Slicing spectrum %s"%source + " out of %s"%len(fluxdata))
        
        Ns   = np.array(Ns_data[source])
        zs   = np.array(zs_data[source])
        ions = np.array(ions_data[source])
        EWs_info = np.array(EW_data[source])

        # Determine wavelength of each absorption line
        lines_obswl = []
        for i, ion in enumerate(ions):
            z_ion = zs[i]
            restwls = ions_dict[ion.astype(str)]
            lines_obswl.append([ion, list(np.array(restwls)*(1.+z_ion))])

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
            #vel_slice = vel[startidx:startidx+num_idxs]

            startidx += slide_idx # number of indices to shift window by

            # Exclude slice if it falls in the Lya forest
            if wave_slice[0] < wl_qso:
                continue

            # Add slice to array of inputs
            fluxslices.append(flux_slice)
            #velslices.append(vel_slice)
            waveslices.append(wave_slice)
            
            lines_present = find_lines_present(wave_slice, lines_obswl, Ns, zs)

            flag, desc, Nline, zline = determine_flag(lines_present, EWs_info, EW_threshold)
            is_abs.append(flag)
            absInfo.append([desc, Nline, zline, "spec_" + str(source)])
 
    chunks = dict(fluxslices = fluxslices, 
                  waveslices = waveslices, 
                  #velslices = velslices, 
                  is_abs = is_abs, 
                  absInfo = absInfo)

    return chunks

def find_lines_present(wave_slice, lines_obswl, Ns, zs):
    """
    Create a list containing all the lines present in the slice along with 
    column densities and redshifts
    
    Input
    -----
    wave_slice: array
        array of wavelengths within slices
    lines_obswl: list
        list of lists containing observed wavelengths of all the lines in the 
        spectrum

    Returns
    -------
    lines_present: list
        ion names, which transitions are present, column density, redshift

    """
    lines_present = []

    for i, line in enumerate(lines_obswl):
        present = []
        for line_wl in line[1]:
            if (line_wl > wave_slice[0]) & (line_wl < wave_slice[-1]):
                present.append(True)
            else:
                present.append(False)
        if True in present:
            lines_present.append([line[0].astype(str),present, Ns[i], zs[i]])
    return lines_present


def determine_flag(lines_present, EWs_info, EW_threshold):
    """
    Return a flag to be used as the target for training
        0: noise - no lines
        1: CIV doublet and nothing else
        2: MgII doublet and nothing else
        3: partial CIV or MgII and nothing else
        4: other single line
        5: other doublets or more
        6: blends

    Input   
    -----
    lines_present: list
        list of strings where each entry corresponds to a line along with 
        column density

    Returns
    -------
    flag: int
    desc: str
        description of line
    Nline: float
        column density, where relevant otherwise 0
    zline: float
        redshift of absorber, where relevant otherwise 0   
    """

    NumIons = len(lines_present)

    Nline = 0
    zline = 0
    
    if NumIons == 0:
        flag = 0 # noise
        desc = '-'
    elif NumIons > 1:
        flag = 6 # blend
        desc = 'blend'
    elif NumIons == 1:
        if lines_present[0][0] == 'CIV':
            if all(lines_present[0][1]):
                #Check that lines are above EW threshold
                #EW_threshold = 0.2
                CIVmask = [name.astype(str) =='CIV' for name in EWs_info['ion']]
                zmask = [str(zcheck)[:5] == str(lines_present[0][3])[:5] for zcheck in EWs_info['zabs']]

                EWs_lines = EWs_info[np.array(CIVmask) & np.array(zmask)]['EW']
                if max(EWs_lines) > EW_threshold:
                    flag = 1
                    desc = 'CIV'
                    Nline = lines_present[0][2]
                    zline = lines_present[0][3]
                else:
                    flag = 7
                    desc = 'weak'
            else:
                flag = 3
                desc ='Partial CIV'
        elif lines_present[0][0] == 'MgII':
            if all(lines_present[0][1]):
                #Check that lines are above EW threshold
                #EW_threshold = 0.05
                CIVmask = [name.astype(str) =='CIV' for name in EWs_info['ion']]
                zmask = [str(zcheck)[:5] == str(lines_present[0][3])[:5] for zcheck in EWs_info['zabs']]

                EWs_lines = EWs_info[np.array(CIVmask) & np.array(zmask)]['EW']
                if max(EWs_lines) > EW_threshold:
                    flag = 2
                    desc = 'MgII'
                    Nline = lines_present[0][2]
                    zline = lines_present[0][3]
                else:
                    flag = 7
                    desc = 'weak'
            else:
                flag = 3
                desc = 'Partial MgII'
        elif np.count_nonzero(lines_present[0][1]) > 1:
            flag = 5
            desc = 'Other doublet'
        else:
            flag = 4
            desc = 'Other single line'

    return flag, desc, Nline, zline

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

def preprocess(chunks):
    """
    Apply preprocessing of data, including balancing the number of samples with 
    each flag so that they are roughly equal. 

    Input   
    -----
    chunks: dict
        contains fluxes, wavelengths and absorber information for all slices

    Returns
    -------
    fluxslices: list
        flux for each slice
    waveslices: list
        wavelength for each slice
    is_abs: list
        flag indicating if slice contains an absorber
    absInfo: list
        where absorber is present, include information e.g. column density
        redshift
    """
    # Balance samples so that there is roughly the same number of noise vs absorbers
    # 0 = noise, 1 = CIV, 2 = MgII, 3 = partial, 4 = other single line,
    # 5 = other doublets or more, 6 = blends

    fluxslicesAll = chunks['fluxslices']
    #velslicesAll  = chunks['velslices']
    waveslicesAll = chunks['waveslices']
    is_absAll     = chunks['is_abs']
    absInfoAll    = chunks['absInfo']

    # Find number of samples for CIV and MgII
    nCIV = len(np.array(is_absAll)[np.array(is_absAll) == 1])
    nMgII = len(np.array(is_absAll)[np.array(is_absAll) == 2])

    # Choose absorber with smallest sample to balance all others to
    nBalance = min([nCIV,nMgII])
    idxs_to_delete = []

    # Loop over other flags and if there are more samples than minimum of nCIV 
    # or nMgII, delete indices at random so that there are the same number
    for flag in np.unique(is_absAll):
        # Find number of samples with this flag
        nFlag = len(np.array(is_absAll)[np.array(is_absAll) == flag])

        if nFlag > nBalance:
            # Find indices of samples with this flag
            idxs_flag = np.where(np.array(is_absAll) == flag)[0]

            # Randomly select indices to delete
            idxs_to_delete+=list(np.random.choice(idxs_flag, len(idxs_flag) - nBalance, replace=False))
            
    #delete indices from flux, vel, wave, is_abs and absInfo
    fluxslices = list(np.delete(np.array(fluxslicesAll),idxs_to_delete,0))
    #velslices  = list(np.delete(np.array(velslicesAll),idxs_to_delete,0))
    waveslices = list(np.delete(np.array(waveslicesAll),idxs_to_delete,0))
    is_abs     = list(np.delete(np.array(is_absAll),idxs_to_delete,0))
    absInfo    = list(np.delete(np.array(absInfoAll),idxs_to_delete,0))

    return fluxslices, waveslices, is_abs, absInfo #velslices, 
 
def run_RF(train, train_isabs):
    """
    Create the random forest model and train it on the training set

    Input
    -----
    train: list
        list of the flux for each slice
    train_isabs: list
        flag indicating whether the slice has an absorber

    Returns
    -------
    model: sklearn.ensemble.forest.RandomForestClassifier
        Random forest model trained on trianing data
    """
    print("Build the forest...")
    Forest=RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,
                                  min_samples_split=10,min_samples_leaf=1,max_features=40,
                                  max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                  n_jobs=40,random_state=120,verbose=0,class_weight='balanced')
    print("Fit training sample...")
    model=Forest.fit(train,train_isabs)

    return model

def read_json_spec(filename):
    with open('NMFPM_data/' + filename) as json_file:
        data = json.load(json_file)
    return data

def split_samples(specDict):
    
    trainSpec = dict()
    testSpec = dict()

    idx_split = int(len(specDict['Flux'])*0.7)

    keys = specDict.keys()

    for key in keys:
        if key == 'wave':
            trainSpec[key] = (specDict[key])
            testSpec[key] = (specDict[key])
        else:
            trainSpec[key] = (specDict[key])[:idx_split]
            testSpec[key] = (specDict[key])[idx_split:]

    return trainSpec, testSpec

def read_hdf5_spec(filelist):

    # Get all files to be used for training
    specfiles = []
    with open(filelist, 'r') as f:
        for line in f:
            specfiles.append(line.strip())

    specfiles.sort()

    fluxdata = []
    Ns_data = []
    zs_data  = []
    ions_data  = []
    z_qso_data = []
    EW_data = []

    # Loop over hdf5 files
    for hdf5file in specfiles:

        print("Working with file: %s"%hdf5file)
        
        # Load file for nsight sightlines
        data = h5py.File(hdf5file, 'r')

        #QSO info
        zems = np.array(data['z_qso']) 
        Rqsos = np.array(data['R_qso']) 

        #Spectra
        wave = np.array(data['wave_weavify'][0])
        fluxes = np.array(data['flux_weavify']) #Of shape nsight x npix

        #convert to velocity-space, relative to wave[0] which is 3700A
        vel = [0]
        for w in range(1,len(wave)):
            wavestep = (wave[w]-wave[w-1])/wave[w]
            velstep = wavestep *_c
            vel.append(vel[w-1] + velstep)
            
        #Column densities
        Ncat = data['absorbers'] #Group of nsight numpy recarrays
        EWcat = data['EWs']

        # Loop over each sightline
        for ind in range(len(Ncat)):
            
            sight = '%s'%ind
            
            # Get qso properties
            zqso = zems[ind]
            Rqso = Rqsos[ind]


            EWsight = np.array(EWcat[sight])
            """            
            EWcat = None
            if 'EWs' in data:
	            EWcat = data['EWs']
	            #Load the EWs for the sightline of interest
	            EWsight = np.array(EWcat[sight])
	            #extract the information for each absorption line created
	            ions = EWsight['ion'] # array of encoded ion strings (e.g. b'MgII')
	            wls = EWsight['wl'] # array of encoded wavelength strings (e.g. b'2796')
	            zabs = EWsight['zabs'] #Array of the redshfit (float) of the absorber
	            lams = EWsight['lam'] #Array of the observed-frame wavelength of the line (float)
	            EWs = EWsight['EW'] #NMFPM rest-frame EW of the absorption line created
	            #print(EWsight)
            """
            # Get flux
            flux = fluxes[ind, :]

            zdata, Ndata, ionsdata  =  extract_abs_properties(ind, sight, Ncat, wave)

            # Drop systems where flux is all -999
            flux_check =  [flux[i] == -999. for i in range(len(flux))]
            if not all(flux_check):             
                fluxdata.append(list(flux))
                Ns_data.append(list(Ndata))
                zs_data.append(list(zdata))
                ions_data.append(list(ionsdata))
                z_qso_data.append(zqso)
                EW_data.append(EWsight)
    
    specDict = dict(Flux  = fluxdata,
                     zdata = zs_data,
                     Ndata = Ns_data,
                     ionsdata = ions_data,
                     wave  = list(wave),
                     orig_wave = data['wave'],
                     vel   = list(vel),
                     zqso  = z_qso_data,
                     EWdata   = EW_data)
    
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

    return zabs, logNs, ions 

########################################################

start_time = datetime.now()

# Read in spectra, with noise, from prepare_data.py
#training_listSpec = read_json_spec('trainSpec_short3.json')
#analysis_listSpec = read_json_spec('analysisSpec.json') # this is the final test spec to 
                                            # compare with other methods, don't 
                                            # confuse with the validation test 
                                            # set used here to check model
training_list = "training_mocks.lst"
training_listSpec = read_hdf5_spec(training_list)

orig_wave_arr = np.array(training_listSpec['orig_wave'])
sample_size = len(training_listSpec['Flux'])

print("Splitting train and test samples...")
trainSpec, testSpec = split_samples(training_listSpec)

print("Slicing spectra...")
# Split spectra into chunks and assign flag for absorbers
# Use a fine sliding for train sample but larger for test to avoid duplication
EW_threshold = 0.2
trainChunks = slice_input(trainSpec,orig_wave_arr,5, EW_threshold)#, wave, vel, 5) # Give all the data and a 
                                                # value to shift the window by
testChunks = slice_input(testSpec,orig_wave_arr,50, EW_threshold)# , wave, vel, 50)
testFineChunks = slice_input(testSpec,orig_wave_arr,5) #, wave, vel, 5)

if __name__ == "__main__":
    print("Preprocessing data...")
    train, train_wave, train_isabs, train_absInfo = preprocess(trainChunks) # train_vel,
    test, test_wave = testChunks['fluxslices'],  testChunks['waveslices'] #test_vel = testChunks['velslices'],
    test_isabs, test_absInfo = testChunks['is_abs'], testChunks['absInfo']

    testFine, testFine_vel, testFine_wave = testFineChunks['fluxslices'], testFineChunks['velslices'], testFineChunks['waveslices'] 
    testFine_isabs, testFine_absInfo = testFineChunks['is_abs'], testFineChunks['absInfo']

    print("Runnning Random Forest...")
    model = run_RF(train, train_isabs)


print("Saving model...")
import joblib
joblib.dump(model, "model_spec" + str(sample_size) + "_EW" + str(EW_threshold) + "_withWeakFlag.joblib")

print("Making predictions...")
# Classify whether test sample are absorber or not
preds = model.predict(test)
#predsFine = model.predict(testFine)

# If you want confidence, return probability of classes
preds_probability = model.predict_proba(test)

#predsFine_probability = model.predict_proba(testFine)
# Return the mean accuracy on the given test data and labels
score = model.score(test,test_isabs)
#scoreFine = model.score(testFine, testFine_isabs)

# Find predictions if only accept doublets which have been identified at 
# high confidence
for prob_cut in [0.25,0.3,0.5]:
    preds_highConf = np.zeros(len(preds))

    strongCIV = [((probs[1]>prob_cut) & (probs[1] == max(probs))) for probs in preds_probability]
    strongMgII = [((probs[2]>prob_cut) & (probs[2] == max(probs))) for probs in preds_probability]
    strongOther = [((max(probs[3:])>prob_cut) & (max(probs[3:]) == max(probs))) for probs in preds_probability]

    preds_highConf[strongCIV] = 1
    preds_highConf[strongMgII] = 2
    preds_highConf[strongOther] = 3

    print("Creating recovery fraction plot for detection of metal types...")
    pl.plotRecoveryFraction_type(preds,test_absInfo, sample_size)
    pl.plotRecoveryFraction_type(preds_highConf,test_absInfo, str(sample_size) + "_highConf_"+str(prob_cut))

    print("Creating identification plots...")
    pl.plotIdentifications(test_isabs,preds,test_absInfo, sample_size)

    print("Plotting confusion matrix...")
    def plotCM(preds, test_isabs, sample_size):
        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        from matplotlib.colors import SymLogNorm
        preds_grouped = np.array(preds)
        preds_grouped[preds_grouped >= 3] = 3
        test_isabs_grouped = np.array(test_isabs)
        test_isabs_grouped[test_isabs_grouped >= 3] = 3
        cm = confusion_matrix(preds_grouped, test_isabs_grouped)
        sn.heatmap(cm, 
                   annot=True, 
                   norm=SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-1.0, 
                                   vmax=1e5, base=10),
                   cbar_kws={"ticks":[0,1,10,1e2,1e3,1e4]}, 
                   annot_kws={"size":8}, 
                   fmt='g')
        plt.xlabel('True Class', fontsize=10)
        plt.ylabel('Predicted Class', fontsize=10)
        plt.savefig("plots/cm_classifier_spec" + str(sample_size) + ".pdf")
        plt.close()
        
        return

    plotCM(preds, test_isabs, sample_size)
    plotCM(preds_highConf, test_isabs, str(sample_size) + "_highConf_"+str(prob_cut))

print("Saving data...")
utl.saveTrainData(train, train_isabs, train_absInfo, train_wave, "train_data" + str(sample_size) + "_EW" + str(EW_threshold) + "_withWeakFlag.pkl") #, train_vel,
utl.saveTestData(test, test_isabs, test_absInfo, test_wave, preds, 
             preds_probability, preds_highConf,"test_data" + str(sample_size) + "_EW" + str(EW_threshold) + "_withWeakFlag.pkl") #, test_vel,


utl.saveTestData(testFine, testFine_isabs, testFine_absInfo, testFine_vel, 
             testFine_wave, predsFine, predsFine_probability, 
             "testFine_data.pkl")

end_time = datetime.now()
runtime = end_time - start_time

print("Runtime (hours) = " + str(runtime.total_seconds()/3600.))

