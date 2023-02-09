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

_c = const.c/1000

def read_spectra(datapath,target_snr):
    #code based on read_NMFPM_spectra.py from Trystyn Berg

    #Noise is the standard deviation in the flux about the continuum
    noise = 1.0/target_snr
    #The percent variation in the noise for each pixel
    noise_scale = 0.1

    #Get all spectra files from stored directory
    test_specs = glob.glob(datapath + 'nmf_pm_MgII_CIV_fake_*')
    fluxdata = []

    Ns_CIV_data, zs_CIV_data = [], []
    Ns_MgII_data, zs_MgII_data = [], []

    #loop over files and read fluxes into single numpy array
    for specfile in test_specs:

        data = np.loadtxt(specfile)
        wave = data[:,0]
        
        #convert to velocity-space, relative to wave[0] which is 3700A
        vel = [0]
        for w in range(1,len(wave)):
            wavestep = (wave[w]-wave[w-1])/wave[w]
            velstep = wavestep *_c
            vel.append(vel[w-1] + velstep)

        #Apply the noise to the flux
        error = np.random.normal(loc=noise, scale=noise_scale*noise, size=len(wave))
        flux = np.random.normal(loc=data[:,1], scale=error) 
        
        fluxdata.append(flux)

        #Extract the input column density/redshift of the absorber(s)
        #Read the header, and split based on ;-delimited info
        f=open(specfile,'r')
        header=f.readlines()[0][:-1].split(';')
        f.close()

        #Create some temporary storage variables
        tmp = []
        NMgII = None
        NCIV = None
        zMgII = None
        zCIV = None

        #Loop through the :-delimited headers and extract the lists for each ion and column(N)/redshift(z)
        for val in header:
            if ':' in val:
                [tag, tmp] = val.split(':')
                if 'CIV' in tag and 'N' in tag: NCIV = tmp
                if 'CIV' in tag and 'z' in tag: zCIV = tmp
                if 'MgII' in tag and 'N' in tag: NMgII = tmp
                if 'MgII' in tag and 'z' in tag: zMgII = tmp

        #Depending if MgII is present or not, extract log column densities and redshifts
        if NMgII is not None:
            try:
                Ns_MgII = np.array(NMgII.strip('[').strip(']').split(',')).astype(float)
                zs_MgII = np.array(zMgII.strip('[').strip(']').split(',')).astype(float)
            except:
                Ns_MgII=np.array([])
                zs_MgII=np.array([])

        #Otherwise create empty array.
        else:
            Ns_MgII=np.array([])
            zs_MgII=np.array([])

        #Same deal, for CIV instead
        if NCIV is not None:
            try:
                Ns_CIV = np.array(NCIV.strip('[').strip(']').split(',')).astype(float)
                zs_CIV = np.array(zCIV.strip('[').strip(']').split(',')).astype(float)
            except:
                Ns_CIV=np.array([])
                zs_CIV=np.array([])
        else:
            Ns_CIV=np.array([])
            zs_CIV=np.array([])

        Ns_CIV_data.append(Ns_CIV)
        zs_CIV_data.append(zs_CIV)
        Ns_MgII_data.append(Ns_MgII)
        zs_MgII_data.append(zs_MgII)
    
    return fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data 

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
	result = [np.lib.stride_tricks.sliding_window_view(i, window_shape = L)[::S] for i in a]
	result_r = np.array(result).reshape(np.shape(result)[0]*np.shape(result)[1],np.shape(result)[2])
	return result_r


def slice_input(data, wave, vel, slide_idx):
    
    fluxdata     = data['Flux']
    Ns_CIV_data  = data['NCIV']
    zs_CIV_data  = data['zCIV']
    Ns_MgII_data = data['NMgII']
    zs_MgII_data = data['zMgII']

    #Slice spectrum into small regions and add tag for whether there is/isnt an absorber
    
    fluxslices = []
    velslices = []
    waveslices = []

    is_abs = []    
    absInfo = []    #will contain logNs and redshifts of absorbers

    for source in range(len(fluxdata)):

        Ns_CIV = Ns_CIV_data[source]
        Ns_MgII = Ns_MgII_data[source]

        zs_CIV = zs_CIV_data[source]
        zs_MgII = zs_MgII_data[source]

        spec = fluxdata[source]

        # determine observed wavelengths of absorbers
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

            #record if there is an absorber or not
            CIV1548_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_CIV_1548_wave]
            CIV1550_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_CIV_1550_wave]    

            MgII2796_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_MgII_2796_wave]    
            MgII2803_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_MgII_2803_wave]  

            #add slice to array of inputs
            fluxslices.append(flux_slice)
            velslices.append(vel_slice)
            waveslices.append(wave_slice)
            
            #if there is no absorption present, flag it as 0
            allLines_present = CIV1548_present + CIV1550_present + MgII2796_present + MgII2803_present
            if True not in allLines_present:
                is_abs.append(0)
                absInfo.append(['-',0, 0, "spec_" + str(source)]) 
                continue

            #check if there are multiple of the same line within the window
            if (sum(CIV1548_present) > 1) |(sum(CIV1550_present) > 1) | (sum(MgII2796_present) > 1) |(sum(MgII2803_present) > 1):
                is_abs.append(4)
                absInfo.append(['multiple of same',0, 0, "spec_" + str(source)]) 
                continue
                
            elif (True in CIV1548_present + CIV1550_present) & (True in MgII2796_present + MgII2803_present):
                is_abs.append(5)
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



def preprocess(chunks):
    #balance samples so that there is roughly the same number of noise vs absorbers

    fluxslicesAll = chunks['fluxslices']
    velslicesAll  = chunks['velslices']
    waveslicesAll = chunks['waveslices']
    is_absAll     = chunks['is_abs']
    absInfoAll    = chunks['absInfo']

    #find number of e.g. CIV absorbers
    nCIV = len(np.array(is_absAll)[np.array(is_absAll) == 1])

    #find where the noise samples are
    idxs_noise = np.where(np.array(is_absAll)==0)[0]

    #check there are more noise chunks than CIV
    if len(idxs_noise) > nCIV:

        #randomly select nCIV indices from idxs_noise
        idxs_to_delete= np.random.choice(idxs_noise, len(idxs_noise) - nCIV, replace=False)

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
 
def run_RF(train, train_isabs, test, test_isabs):

	print("Build the forest...")
    Forest=RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,
                                  min_samples_split=10,min_samples_leaf=1,max_features=40,
                                  max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                  n_jobs=40,random_state=120,verbose=0,class_weight='balanced')
	print("Fit training sample...")
    model=Forest.fit(train,train_isabs)

    return model



def plotRecoveryFraction_type(test_isabs,preds,test_absInfo):

    binsize = 0.3
    logNbins = np.arange(12,15.75,binsize)
    Nvals = np.array(test_absInfo)[:,1]
    Ntype = np.array(test_absInfo)[:,0]

    recoveryFracs_CIV = []
    recoveryFracsErr_CIV = []
    recoveryFracs_MgII = []
    recoveryFracsErr_MgII = []

    recoveryFracs_partial = []
    recoveryFracsErr_partial = []

    for n in range(1,len(logNbins)):

        minN = logNbins[n-1]
        maxN = logNbins[n]

        Nmask = [(float(nv) > minN) & (float(nv) <= maxN) for nv in Nvals]

        CIVmask = [t == 'CIV' for t in Ntype]
        MgIImask = [t == 'MgII' for t in Ntype]
        partialMgIImask = [t == 'partial MgII' for t in Ntype]
        partialCIVmask = [t == 'partial CIV' for t in Ntype]

        preds_bin_CIV = preds[np.array(Nmask) & np.array(CIVmask)]
        preds_bin_MgII = preds[np.array(Nmask) & np.array(MgIImask)]
        preds_bin_partialCIV = preds[np.array(Nmask) & np.array(partialCIVmask)]
        preds_bin_partialMgII = preds[np.array(Nmask) & np.array(partialMgIImask)]        

        #number of true absorbers is the length of the array
        true_abs_CIV = float(len(preds_bin_CIV))
        recovered_abs_CIV = float(len(preds_bin_CIV[preds_bin_CIV == 1]))
        true_abs_MgII = float(len(preds_bin_MgII))
        recovered_abs_MgII = float(len(preds_bin_MgII[preds_bin_MgII == 2]))

        #partial absorption
        true_abs_partialCIV = float(len(preds_bin_partialCIV))
        recovered_abs_partialCIV = float(len(preds_bin_partialCIV[preds_bin_partialCIV == 3]))
        true_abs_partialMgII = float(len(preds_bin_partialMgII))
        recovered_abs_partialMgII = float(len(preds_bin_partialMgII[preds_bin_partialMgII == 3]))

        #calculate recovery fraction
        CIVfrac,CIVfracerr = calc_recovery(true_abs_CIV, recovered_abs_CIV)
        recoveryFracs_CIV.append(CIVfrac)
        recoveryFracsErr_CIV.append(CIVfracerr)

        MgIIfrac,MgIIfracerr = calc_recovery(true_abs_MgII , recovered_abs_MgII)
        recoveryFracs_MgII.append(MgIIfrac)
        recoveryFracsErr_MgII.append(MgIIfracerr)

        partialfrac,partialfracerr = calc_recovery(true_abs_partialMgII + true_abs_partialCIV, recovered_abs_partialMgII + recovered_abs_partialCIV)
        recoveryFracs_partial.append(partialfrac)
        recoveryFracsErr_partial.append(partialfracerr)

    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_MgII,yerr=recoveryFracsErr_MgII,xerr=binsize/2,linestyle=' ',capsize=3,label='MgII')
    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_CIV,yerr=recoveryFracsErr_CIV,xerr=binsize/2,linestyle=' ',capsize=3,label='CIV')
    #plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_partial,yerr=recoveryFracsErr_partial,xerr=binsize/2,linestyle=' ',capsize=3,label='partial')

    plt.legend()
    plt.ylim(0,1.1)
    plt.title('Identifying correct metal')
    plt.xlabel('logN')
    plt.ylabel('Recovery Fraction')
    plt.savefig('plots/rf_2.pdf')
    plt.close()

    return

def calc_recovery(true, recovered):
    if true == 0:
        return -999,-999
    elif recovered == 0:
        return 0,0
    else:
        frac = recovered/true
        fracerr = frac * np.sqrt((recovered/recovered**2)+(true/true**2))

    return frac, fracerr

def plotIdentifications(test_isabs,preds,test_absInfo):
    """
    Plot histograms of number of true absorbers, number correctly identified and those identified but as the wrong absorber

    """

    binsize = 0.3
    logNbins = np.arange(12,15.75,binsize)
    Nvals = np.array(test_absInfo)[:,1]
    Ntype = np.array(test_absInfo)[:,0]

    Total_CIV = []
    Correct_CIV = []
    MisIdentified_CIV = []

    Total_MgII = []
    Correct_MgII = []
    MisIdentified_MgII = []

    for n in range(1,len(logNbins)):

        minN = logNbins[n-1]
        maxN = logNbins[n]

        Nmask = [(float(nv) > minN) & (float(nv) <= maxN) for nv in Nvals]

        CIVmask = [t == 'CIV' for t in Ntype]
        MgIImask = [t == 'MgII' for t in Ntype]

        preds_bin_CIV = preds[np.array(Nmask) & np.array(CIVmask)]
        preds_bin_MgII = preds[np.array(Nmask) & np.array(MgIImask)]
       
        Total_CIV.append(float(len(preds_bin_CIV)))
        Correct_CIV.append(float(len(preds_bin_CIV[preds_bin_CIV == 1])))
        MisIdentified_CIV.append(float(len(preds_bin_CIV[preds_bin_CIV == 2])))

        Total_MgII.append(float(len(preds_bin_MgII)))
        Correct_MgII.append(float(len(preds_bin_MgII[preds_bin_MgII == 2])))
        MisIdentified_MgII.append(float(len(preds_bin_MgII[preds_bin_MgII == 1])))

    #Plot CIV results
    plt.fill_between(logNbins[:-1]+binsize/2,Total_CIV,np.array(Correct_CIV) + np.array(MisIdentified_CIV),color='k',label='All CIV', step="pre", alpha=0.2)
    plt.fill_between(logNbins[:-1]+binsize/2,Correct_CIV,color='g',label='Correctly identified as CIV', step="pre", alpha=0.4)
    plt.fill_between(logNbins[:-1]+binsize/2,np.array(Correct_CIV) + np.array(MisIdentified_CIV),Correct_CIV,color='r',label='Plus Incorrectly identified as MgII', step="pre", alpha=0.4)

    plt.legend()
    plt.xlabel('logN')
    plt.ylim(0,np.max(Total_CIV)*1.3)

    plt.ylabel('Number of absorbers')
    plt.savefig('plots/idents_1.pdf')
    plt.close()

    #Plot MgII results
    plt.fill_between(logNbins[:-1]+binsize/2,Total_MgII,np.array(Correct_MgII) + np.array(MisIdentified_MgII),color='k',label='All MgII', step="pre", alpha=0.2)
    plt.fill_between(logNbins[:-1]+binsize/2,Correct_MgII,color='g',label='Correctly identified as MgII', step="pre", alpha=0.4)
    plt.fill_between(logNbins[:-1]+binsize/2,np.array(Correct_MgII) + np.array(MisIdentified_MgII),Correct_MgII,color='r',label='Plus Incorrectly identified as CIV', step="pre", alpha=0.4)

    plt.legend()
    plt.xlabel('logN')
    plt.ylim(0,np.max(Total_MgII)*1.2)

    plt.ylabel('Number of absorbers')
    plt.savefig('plots/idents_2.pdf')
    plt.close()

    return

########################################################
#scriptpath = os.path.dirname(os.path.abspath(__file__))
datapath = "../data/NMFPM_data/" #scriptpath + "/NMFPM_data/"
print(datapath)
print("Reading spectra and adding noise...")

#Set the target S/N in the continuum for the spectra
target_snr = 5.0
fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data = read_spectra(datapath,target_snr)

print("Split train-test samples...")
trainSpec, testSpec = split_samples(fluxdata, wave, vel, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data)

print("Slicing spectra...")
#split spectra into chunks and assign flag for asborbers
#use a fine sliding for train sample but larger for test to avoid duplication
trainChunks = slice_input(trainSpec, wave, vel, 5) #give all the data and a value to shift the window by
testChunks = slice_input(testSpec, wave, vel, 50)
testFineChunks = slice_input(testSpec, wave, vel, 5)

if __name__ == "__main__":
        
    print("Preprocessing data...")
    # e.g. balance samples
    train, train_vel, train_wave, train_isabs, train_absInfo = preprocess(trainChunks)
    test, test_vel, test_wave = testChunks['fluxslices'], testChunks['velslices'], testChunks['waveslices'] 
    test_isabs, test_absInfo = testChunks['is_abs'], testChunks['absInfo']

    testFine, testFine_vel, testFine_wave = testFineChunks['fluxslices'], testFineChunks['velslices'], testFineChunks['waveslices'] 
    testFine_isabs, testFine_absInfo = testFineChunks['is_abs'], testFineChunks['absInfo']

    print("Runnning Random Forest...")
    model = run_RF(train, train_isabs, test, test_isabs)

print("Predictions...")
#classify whether test sample are absorber or not
preds = model.predict(test)
#if you want confidence, return probability of classes
preds_probability = model.predict_proba(test)
#Return the mean accuracy on the given test data and labels
score = model.score(test,test_isabs)

print("Creating recovery fraction plot for detection of metal types...")
plotRecoveryFraction_type(test_isabs,preds,test_absInfo)

print("Creating identification plots...")
plotIdentifications(test_isabs,preds,test_absInfo)

#output data
import pandas as pd

d = {'flux': train,
     'isabs': train_isabs, 
     'absInfo': train_absInfo, 
     'vel': train_vel,
     'wave': train_wave}

df = pd.DataFrame(data=d)
df.to_pickle("train_data.pkl")

d = {'flux': test,
     'isabs': test_isabs, 
     'absInfo': test_absInfo, 
     'vel': test_vel,
     'wave':test_wave}

df = pd.DataFrame(data=d)
df.to_pickle("test_data.pkl")

d = {'flux': test,
     'isabs': test_isabs, 
     'absInfo': test_absInfo, 
     'vel': test_vel,
     'wave':test_wave}

df = pd.DataFrame(data=d)
df.to_pickle("test_data.pkl")

d = {'flux': testFine,
     'isabs': testFine_isabs, 
     'absInfo': testFine_absInfo, 
     'vel': testFine_vel,
     'wave':testFine_wave}

df = pd.DataFrame(data=d)
df.to_pickle("testFine_data.pkl")

