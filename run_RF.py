from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.decomposition import NMF 


def read_spectra(datapath):
    #code based on read_NMFPM_spectra.py from Trystyn Berg

    #Set the target S/N in the continuum for the spectra
    target_snr = 10.0
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
        flux = np.random.normal(loc=data[:,1], scale=error) # !!! Trying with noise free spectra first! remember to change back data[:,1] #
        
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

def slice_input(fluxdata,wave, zs_CIV_data, zs_MgII_data):
    #Slice spectrum into small regions and add tag for whether there is/isnt an absorber
    
    fluxslices = []
    waveslices = []
    is_abs = []

    for source in range(len(fluxdata)):
        spec = fluxdata[source]

        #indexs to split spectrum into
        idxs = np.arange(0,len(spec),200)    
        
        # determine observed wavelengths of absorbers
        obs_CIV_1548 = 1548*(zs_CIV_data[source] + 1)
        obs_MgII_2796 = 2796*(zs_MgII_data[source] + 1)
        
        # for any kind of absorber
        obs_absorber = np.concatenate([obs_CIV_1548,obs_MgII_2796])

        for i in range(1,len(idxs-1)):
            flux_slice = spec[idxs[i-1]:idxs[i]] 
            fluxslices.append(flux_slice)
            wave_slice = wave[idxs[i-1]:idxs[i]]
            waveslices.append(wave_slice)

            #record if there is an absorber or not
            absorber_present = [(wl > wave_slice[0]) & (wl < wave_slice[-1]) for wl in obs_absorber]
            
            if True in absorber_present:
                is_abs.append(1)
            else:
                is_abs.append(0)

    return fluxslices,waveslices,is_abs


def preprocess(fluxslices, waveslices,is_abs):
    
    ###
    #put preprocessing,e.g. scaling steps here
    ###

    # Train-test split
    idx_split = len(fluxslices)/2

    train = fluxslices[:idx_split]
    train_isabs = is_abs[:idx_split]
    test = fluxslices[idx_split:]
    test_isabs = is_abs[idx_split:]

    return train, train_isabs, test, test_isabs

def run_RF(train, train_isabs, test, test_isabs):

    #build the forest max_features=400 for no binning
    Forest=RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,
                                  min_samples_split=4,min_samples_leaf=1,max_features=40,
                                  max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                  n_jobs=40,random_state=120,verbose=0,class_weight='balanced')

    model=Forest.fit(train,train_isabs)

    return model

datapath = "/home/emma/Documents/WEAVE/data/NMFPM_data/"

fluxdata, wave, Ns_CIV_data, zs_CIV_data, Ns_MgII_data, zs_MgII_data = read_spectra(datapath)

fluxslices, waveslices,is_abs = slice_input(fluxdata,wave, zs_CIV_data, zs_MgII_data)

nabs = len(np.array(is_abs)[np.array(is_abs)==1])
nempty = len(np.array(is_abs)[np.array(is_abs)==0])

train, train_isabs, test, test_isabs = preprocess(fluxslices, waveslices,is_abs )

model = run_RF(train, train_isabs, test, test_isabs)

#classify absorber or not
preds = model.predict(test)

test_isabs=np.array(test_isabs)

# Number of True absorber predicted to be absorber
isAbs_and_predAbs = np.where(( test_isabs == 1) & (preds==1))

# Number of True absorber not predicted to be absorber
isAbs_and_NotpredAbs = np.where(( test_isabs == 1) & (preds!=1))

# Number of not absorber predicted to be absorber
NotAbs_and_predAbs = np.where(( test_isabs != 1) & (preds==1))

# Number of not absorber predicted to not be absorber
NotAbs_and_NotpredAbs = np.where(( test_isabs != 1) & (preds!=1))


print(len(isAbs_and_predAbs[0]),len(isAbs_and_NotpredAbs[0]),len(NotAbs_and_predAbs[0]),len(NotAbs_and_NotpredAbs[0]))



