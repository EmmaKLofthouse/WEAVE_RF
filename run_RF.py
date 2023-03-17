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
    absInfo = [] # will contain logNs and redshifts of absorbers

    for source in range(len(fluxdata)):
        if source%100 == 0:
            print("Slicing spectrum %s"%source + " out of %s"%len(fluxdata))
        
        Ns_CIV = np.array(Ns_CIV_data[source])
        Ns_MgII = np.array(Ns_MgII_data[source])

        zs_CIV = np.array(zs_CIV_data[source])
        zs_MgII = np.array(zs_MgII_data[source])

        spec = fluxdata[source]

        # Determine observed wavelengths of absorbers
        obs_CIV_1548_wave = 1548.*(zs_CIV + 1)
        obs_CIV_1550_wave = 1550.*(zs_CIV + 1)

        obs_MgII_2796_wave = 2796.4*(zs_MgII + 1)
        obs_MgII_2803_wave = 2803.5*(zs_MgII + 1)
       
        startidx = 0 #initialise first chunk
        num_idxs = 100 #size of window

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
            fluxslices.append(flux_slice)
            velslices.append(vel_slice)
            waveslices.append(wave_slice)
            
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
                        absInfo.append(['CIV', Ns_CIV[matchidx[0]], zs_CIV[matchidx[0]], "spec_" + str(source)])
                        continue
                is_abs.append(3)
                absInfo.append(['partial CIV',0,0, "spec_" + str(source)])
                continue

            elif ((True in MgII2796_present) | (True in MgII2803_present)): 
                matchidx = np.where(MgII2796_present)[0]    
                if len(matchidx) > 0:
                    if MgII2803_present[matchidx[0]]:
                        is_abs.append(2)
                        absInfo.append(['MgII', Ns_MgII[matchidx[0]], zs_MgII[matchidx[0]], "spec_" + str(source)])
                        continue
                is_abs.append(3)
                absInfo.append(['partial MgII',0,0, "spec_" + str(source)])
                
    chunks = dict(fluxslices = fluxslices, 
                  waveslices = waveslices, 
                  velslices = velslices, 
                  is_abs = is_abs, 
                  absInfo = absInfo)

    return chunks

def preprocess(chunks):
    #balance samples so that there is roughly the same number of noise vs absorbers

    fluxslicesAll = chunks['fluxslices']
    velslicesAll  = chunks['velslices']
    waveslicesAll = chunks['waveslices']
    is_absAll     = chunks['is_abs']
    absInfoAll    = chunks['absInfo']

    #find number of e.g. CIV absorbers
    nCIV = len(np.array(is_absAll)[np.array(is_absAll) == 1])
    npartial = len(np.array(is_absAll)[np.array(is_absAll) == 3])

    #find where the noise samples are
    idxs_noise = np.where(np.array(is_absAll)==0)[0]

    #check there are more noise chunks than CIV
    if len(idxs_noise) > nCIV:

        #randomly select nCIV indices from idxs_noise
        idxs_to_delete = np.random.choice(idxs_noise, len(idxs_noise) - nCIV, replace=False)
        #idxs_to_delete = np.random.choice(idxs_noise, len(idxs_noise) - npartial, replace=False)
        
        #delete indices from flux, vel, wave, is_abs and absInfo
        fluxslices = list(np.delete(np.array(fluxslicesAll),idxs_to_delete,0))
        velslices  = list(np.delete(np.array(velslicesAll),idxs_to_delete,0))
        waveslices = list(np.delete(np.array(waveslicesAll),idxs_to_delete,0))
        is_abs     = list(np.delete(np.array(is_absAll),idxs_to_delete,0))
        absInfo    = list(np.delete(np.array(absInfoAll),idxs_to_delete,0))

    else:
        fluxslices = fluxslicesAll
        velslices  = velslicesAll
        waveslices = waveslicesAll
        is_abs     = is_absAll
        absInfo    = absInfoAll

    return fluxslices, velslices, waveslices, is_abs, absInfo
 
def run_RF(train, train_isabs):

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

    fluxdata = specDict['Flux'] 
    Ns_CIV_data = specDict['NCIV'] 
    zs_CIV_data = specDict['zCIV'] 
    Ns_MgII_data = specDict['NMgII']
    zs_MgII_data = specDict['zMgII'] 

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

start_time = datetime.now()

# Read in spectra, with noise, from prepare_data.py
training_listSpec = read_json_spec('trainSpec.json')
#test_listSpec = read_json_spec('testSpec.json') # this is the final test spec to 
                                            # compare with other methods, don't 
                                            # confuse with the validation test 
                                            # set used here to check model

wave = training_listSpec['wave']
vel = training_listSpec['vel']

trainSpec, testSpec = split_samples(training_listSpec)

print("Slicing spectra...")
# Split spectra into chunks and assign flag for absorbers
# Use a fine sliding for train sample but larger for test to avoid duplication
trainChunks = slice_input(trainSpec, wave, vel, 5) # Give all the data and a 
                                                # value to shift the window by
testChunks = slice_input(testSpec, wave, vel, 50)
testFineChunks = slice_input(testSpec, wave, vel, 5)

if __name__ == "__main__":
    print("Preprocessing data...")
    # Balance samples
    train, train_vel, train_wave, train_isabs, train_absInfo = preprocess(trainChunks)
    test, test_vel, test_wave = testChunks['fluxslices'], testChunks['velslices'], testChunks['waveslices'] 
    test_isabs, test_absInfo = testChunks['is_abs'], testChunks['absInfo']

    testFine, testFine_vel, testFine_wave = testFineChunks['fluxslices'], testFineChunks['velslices'], testFineChunks['waveslices'] 
    testFine_isabs, testFine_absInfo = testFineChunks['is_abs'], testFineChunks['absInfo']

    print("Runnning Random Forest...")
    model = run_RF(train, train_isabs)

print("Making predictions...")
# Classify whether test sample are absorber or not
preds = model.predict(test)
predsFine = model.predict(testFine)

# If you want confidence, return probability of classes
preds_probability = model.predict_proba(test)
predsFine_probability = model.predict_proba(testFine)
# Return the mean accuracy on the given test data and labels
score = model.score(test,test_isabs)
scoreFine = model.score(testFine, testFine_isabs)

print("Creating recovery fraction plot for detection of metal types...")
pl.plotRecoveryFraction_type(test_isabs,preds,test_absInfo)

print("Creating identification plots...")
pl.plotIdentifications(test_isabs,preds,test_absInfo)

print("Plotting confusion matrix...")

def plotCM(preds, test_isabs):
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    from matplotlib.colors import SymLogNorm
    preds_grouped = np.array(preds)
    preds_grouped[preds_grouped >= 4] = 4
    test_isabs_grouped = np.array(test_isabs)
    test_isabs_grouped[test_isabs_grouped >= 4] = 4
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
    plt.savefig("cm_classifier.pdf")
    plt.close()
    
    return
plotCM(preds, test_isabs)

print("Saving data...")
utl.saveTrainData(train, train_isabs, train_absInfo, train_vel, train_wave, 
              "train_data.pkl")
utl.saveTestData(test, test_isabs, test_absInfo, test_vel, test_wave, preds, 
             preds_probability, "test_data.pkl")
utl.saveTestData(testFine, testFine_isabs, testFine_absInfo, testFine_vel, 
             testFine_wave, predsFine, predsFine_probability, 
             "testFine_data.pkl")

end_time = datetime.now()
runtime = end_time - start_time

print("Runtime (hours) = " + str(runtime.total_seconds()/3600.))

