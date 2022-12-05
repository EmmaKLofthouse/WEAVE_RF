import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.decomposition import NMF 

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

    return fluxdata, wave, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data 

def slice_input(fluxdata,wave, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data ):
    #Slice spectrum into small regions and add tag for whether there is/isnt an absorber
    
    fluxslices = []
    waveslices = []
    is_abs = []
    
    logNs = []
    
    nshow = 0

    for source in range(len(fluxdata)):

        Ns_CIV = Ns_CIV_data[source]
        Ns_MgII = Ns_MgII_data[source]

        spec = fluxdata[source]

        #indexs to split spectrum into
        idxs = np.arange(0,len(spec),200)    
        
        # determine observed wavelengths of absorbers
        obs_CIV_1548 = 1548*(zs_CIV_data[source] + 1)
        obs_CIV_1550 = 1550*(zs_CIV_data[source] + 1)

        obs_MgII_2796 = 2796.4*(zs_MgII_data[source] + 1)
        obs_MgII_2803 = 2803.5*(zs_MgII_data[source] + 1)
 
        for i in range(1,len(idxs-1)):

            flux_slice = spec[idxs[i-1]:idxs[i]] 
            wave_slice = wave[idxs[i-1]:idxs[i]]

            #record if there is an absorber or not
            CIV1548_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_CIV_1548]
            CIV1550_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_CIV_1550]    

            MgII2796_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_MgII_2796]    
            MgII2803_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_MgII_2803]    

            if (True in CIV1548_present) or (True in CIV1550_present) or (True in MgII2796_present) or (True in MgII2803_present):

                #check that the absorber is not cut off, if it is, skip it
                if ((True in CIV1548_present) != (True in CIV1550_present)) | ((True in MgII2796_present) != (True in MgII2803_present)):
                    continue

                # otherwise add to flux and wavelength arrays
                fluxslices.append(flux_slice)
                waveslices.append(wave_slice)

                #record column densities
                if True in CIV1548_present:
                    matchidx = np.where(CIV1548_present)[0][0]       
                    logNs.append(['CIV',Ns_CIV[matchidx]])
                    is_abs.append(1)
                elif True in MgII2796_present:
                    matchidx = np.where(MgII2796_present)[0][0]       
                    logNs.append(['MgII',Ns_MgII[matchidx]])
                    is_abs.append(2)

            else:
                fluxslices.append(flux_slice)
                waveslices.append(wave_slice)

                is_abs.append(0)
                logNs.append(['-',0])
            
            """
            if (nshow <20) and (True in absorber_present):
                plt.plot(wave_slice,flux_slice)
                plt.show()
                plt.close()
                nshow += 1
            """

    return fluxslices, waveslices, is_abs, logNs


def preprocess(fluxslices, waveslices, is_abs, logNs):
    
    ###
    #put preprocessing,e.g. scaling steps here
    ###

    # Train-test split
    idx_split = int(len(fluxslices)/1.5)

    train       = fluxslices[:idx_split]
    train_isabs = is_abs[:idx_split]
    train_logNs = logNs[:idx_split]

    test        = fluxslices[idx_split:]
    test_isabs  = is_abs[idx_split:]
    test_logNs  = logNs[idx_split:]

    return train, train_isabs, test, test_isabs, train_logNs, test_logNs
 
def run_RF(train, train_isabs, test, test_isabs):

    #build the forest 
    Forest=RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,
                                  min_samples_split=10,min_samples_leaf=1,max_features=40,
                                  max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                  n_jobs=40,random_state=120,verbose=0,class_weight='balanced')

    model=Forest.fit(train,train_isabs)

    return model

def plotRecoveryFraction(test_isabs,preds,test_logNs):

    binsize = 0.3
    logNbins = np.arange(12,15.75,binsize)
    Nvals = np.array(test_logNs)[:,1]
    Ntype = np.array(test_logNs)[:,0]
    recoveryFracs_CIV = []
    recoveryFracsErr_CIV = []
    recoveryFracs_MgII = []
    recoveryFracsErr_MgII = []

    for n in range(1,len(logNbins)):

        minN = logNbins[n-1]
        maxN = logNbins[n]

        Nmask = [(float(nv) > minN) & (float(nv) <= maxN) for nv in Nvals]

        CIVmask = [t == 'CIV' for t in Ntype]
        MgIImask = [t == 'MgII' for t in Ntype]

        preds_bin_CIV = preds[np.array(Nmask) & np.array(CIVmask)]
        preds_bin_MgII = preds[np.array(Nmask) & np.array(MgIImask)]
       
        #number of true absorbers is the length of the array
        true_abs_CIV = float(len(preds_bin_CIV))
        recovered_abs_CIV = float(len(preds_bin_CIV[preds_bin_CIV >= 1]))
        true_abs_MgII = float(len(preds_bin_MgII))
        recovered_abs_MgII = float(len(preds_bin_MgII[preds_bin_MgII >= 1]))

        
        if true_abs_CIV == 0:
            recoveryFracs_CIV.append(-999)
            recoveryFracsErr_CIV.append(-999)
        elif recovered_abs_CIV == 0:
            recoveryFracs_CIV.append(0)
            recoveryFracsErr_CIV.append(0)
        else:
            recoveryFracs_CIV.append(recovered_abs_CIV/true_abs_CIV)
            recoveryFracsErr_CIV.append(recovered_abs_CIV/true_abs_CIV * np.sqrt((recovered_abs_CIV/recovered_abs_CIV**2)+(true_abs_CIV/true_abs_CIV**2)))

        if true_abs_MgII == 0:
            recoveryFracs_MgII.append(-999.)
            recoveryFracsErr_MgII.append(-999)
        elif recovered_abs_MgII == 0:
            recoveryFracs_MgII.append(0)
            recoveryFracsErr_MgII.append(0)
        else:
            recoveryFracs_MgII.append(recovered_abs_MgII/true_abs_MgII)
            recoveryFracsErr_MgII.append(recovered_abs_MgII/true_abs_MgII * np.sqrt((recovered_abs_MgII/recovered_abs_MgII**2)+(true_abs_MgII/true_abs_MgII**2)))


    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_MgII,yerr=recoveryFracsErr_MgII,xerr=binsize/2,linestyle=' ',capsize=3,label='MgII')
    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_CIV,yerr=recoveryFracsErr_CIV,xerr=binsize/2,linestyle=' ',capsize=3,label='CIV')
    plt.legend()
    plt.xlabel('logN')
    plt.ylim(0,1.1)
    plt.ylabel('Recovery Fraction')
    plt.show()
    plt.close()

    return

def plotRecoveryFraction_type(test_isabs,preds,test_logNs):

    binsize = 0.3
    logNbins = np.arange(12,15.75,binsize)
    Nvals = np.array(test_logNs)[:,1]
    Ntype = np.array(test_logNs)[:,0]

    recoveryFracs_CIV = []
    recoveryFracsErr_CIV = []
    recoveryFracs_MgII = []
    recoveryFracsErr_MgII = []

    for n in range(1,len(logNbins)):

        minN = logNbins[n-1]
        maxN = logNbins[n]

        Nmask = [(float(nv) > minN) & (float(nv) <= maxN) for nv in Nvals]

        CIVmask = [t == 'CIV' for t in Ntype]
        MgIImask = [t == 'MgII' for t in Ntype]

        preds_bin_CIV = preds[np.array(Nmask) & np.array(CIVmask)]
        preds_bin_MgII = preds[np.array(Nmask) & np.array(MgIImask)]
       
        #number of true absorbers is the length of the array
        true_abs_CIV = float(len(preds_bin_CIV))
        recovered_abs_CIV = float(len(preds_bin_CIV[preds_bin_CIV == 1]))
        true_abs_MgII = float(len(preds_bin_MgII))
        recovered_abs_MgII = float(len(preds_bin_MgII[preds_bin_MgII == 2]))

        
        if true_abs_CIV == 0:
            recoveryFracs_CIV.append(-999)
            recoveryFracsErr_CIV.append(-999)
        elif recovered_abs_CIV == 0:
            recoveryFracs_CIV.append(0)
            recoveryFracsErr_CIV.append(0)
        else:
            recoveryFracs_CIV.append(recovered_abs_CIV/true_abs_CIV)
            recoveryFracsErr_CIV.append(recovered_abs_CIV/true_abs_CIV * np.sqrt((recovered_abs_CIV/recovered_abs_CIV**2)+(true_abs_CIV/true_abs_CIV**2)))

        if true_abs_MgII == 0:
            recoveryFracs_MgII.append(-999.)
            recoveryFracsErr_MgII.append(-999)
        elif recovered_abs_MgII == 0:
            recoveryFracs_MgII.append(0)
            recoveryFracsErr_MgII.append(0)
        else:
            recoveryFracs_MgII.append(recovered_abs_MgII/true_abs_MgII)
            recoveryFracsErr_MgII.append(recovered_abs_MgII/true_abs_MgII * np.sqrt((recovered_abs_MgII/recovered_abs_MgII**2)+(true_abs_MgII/true_abs_MgII**2)))


    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_MgII,yerr=recoveryFracsErr_MgII,xerr=binsize/2,linestyle=' ',capsize=3,label='MgII')
    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_CIV,yerr=recoveryFracsErr_CIV,xerr=binsize/2,linestyle=' ',capsize=3,label='CIV')
    plt.legend()
    plt.ylim(0,1.1)
    plt.title('Identifying correct metal')
    plt.xlabel('logN')
    plt.ylabel('Recovery Fraction')
    plt.show()
    plt.close()

    return

def plotIdentifications(test_isabs,preds,test_logNs):
    """
    Plot histograms of number of true absorbers, number correctly identified and those identified but as the wrong absorber

    """

    binsize = 0.3
    logNbins = np.arange(12,15.75,binsize)
    Nvals = np.array(test_logNs)[:,1]
    Ntype = np.array(test_logNs)[:,0]

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
    #plt.step(logNbins[:-1]+binsize/2,Total_CIV,color='k',label='All CIV')
    #plt.step(logNbins[:-1]+binsize/2,Correct_CIV,color='g',label='Correctly identified \n as CIV')
    #plt.step(logNbins[:-1]+binsize/2,np.array(Correct_CIV) + np.array(MisIdentified_CIV),color='r',label='Plus Incorrectly identified \n as MgII')

    plt.fill_between(logNbins[:-1]+binsize/2,Total_CIV,np.array(Correct_CIV) + np.array(MisIdentified_CIV),color='k',label='All CIV', step="pre", alpha=0.2)
    plt.fill_between(logNbins[:-1]+binsize/2,Correct_CIV,color='g',label='Correctly identified as CIV', step="pre", alpha=0.4)
    plt.fill_between(logNbins[:-1]+binsize/2,np.array(Correct_CIV) + np.array(MisIdentified_CIV),Correct_CIV,color='r',label='Plus Incorrectly identified as MgII', step="pre", alpha=0.4)

    plt.legend()
    plt.xlabel('logN')
    plt.ylim(0,np.max(Total_CIV)*1.3)

    plt.ylabel('Number of absorbers')
    plt.show()
    plt.close()

    #Plot MgII results
    #plt.step(logNbins[:-1]+binsize/2,Total_MgII,color='k',label='All MgII')
    #plt.step(logNbins[:-1]+binsize/2,Correct_MgII,color='g',label='Correctly identified \n as MgII')
    #plt.step(logNbins[:-1]+binsize/2,np.array(Correct_MgII) + np.array(MisIdentified_MgII),color='r',label='Plus Incorrectly identified \n as CIV')


    plt.fill_between(logNbins[:-1]+binsize/2,Total_MgII,np.array(Correct_MgII) + np.array(MisIdentified_MgII),color='k',label='All MgII', step="pre", alpha=0.2)
    plt.fill_between(logNbins[:-1]+binsize/2,Correct_MgII,color='g',label='Correctly identified as MgII', step="pre", alpha=0.4)
    plt.fill_between(logNbins[:-1]+binsize/2,np.array(Correct_MgII) + np.array(MisIdentified_MgII),Correct_MgII,color='r',label='Plus Incorrectly identified as CIV', step="pre", alpha=0.4)


    plt.legend()
    plt.xlabel('logN')
    plt.ylim(0,np.max(Total_MgII)*1.2)

    plt.ylabel('Number of absorbers')
    plt.show()
    plt.close()


    return


########################################################

datapath = "/home/emma/Documents/WEAVE/data/NMFPM_data/"

print("Reading spectra and adding noise...")

#Set the target S/N in the continuum for the spectra
target_snr = 5.0
fluxdata, wave, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data = read_spectra(datapath,target_snr)

print("Slicing spectra...")
fluxslices, waveslices, is_abs, logNs = slice_input(fluxdata,wave,Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data)

nabs = len(np.array(is_abs)[np.array(is_abs)==1])
nempty = len(np.array(is_abs)[np.array(is_abs)==0])

print("Preprocessing data...")
train, train_isabs, test, test_isabs, train_logNs, test_logNs = preprocess(fluxslices, waveslices, is_abs, logNs)

print("Runnning Random Forest...")
model = run_RF(train, train_isabs, test, test_isabs)

print("Predictions...")
#classify whether test sample are absorber or not
preds = model.predict(test)

test_isabs=np.array(test_isabs)

#For absorber/not absorber:
# Number of True absorber predicted to be absorber
isAbs_and_predAbs = np.where(( test_isabs >= 1) & (preds >= 1))

# Number of True absorber not predicted to be absorber
isAbs_and_NotpredAbs = np.where(( test_isabs >= 1) & (preds == 0))

# Number of not absorber predicted to be absorber
NotAbs_and_predAbs = np.where(( test_isabs == 0) & (preds >= 1))

# Number of not absorber predicted to not be absorber
NotAbs_and_NotpredAbs = np.where(( test_isabs == 0) & (preds==0))

print(len(isAbs_and_predAbs[0]),len(isAbs_and_NotpredAbs[0]),len(NotAbs_and_predAbs[0]),len(NotAbs_and_NotpredAbs[0]))

print("Creating recovery fraction plot for any kind of absorption...")
plotRecoveryFraction(test_isabs,preds,test_logNs)


print("Creating recovery fraction plot for detection of metal types...")
plotRecoveryFraction_type(test_isabs,preds,test_logNs)


plotIdentifications(test_isabs,preds,test_logNs)








