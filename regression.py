"""
Run random forest regression to identify redshift of absorbers

"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

def create_regression_plots(preds_z,z_test,preds_idx,target_index_test):

    #dz = abs(np.array(preds_z) - np.array(z_test))
    didx = abs(preds_idx - target_index_test)
    dz = didx *18/3e5

    p = plt.scatter(target_index_test,preds_idx,marker='.',c=dz)
    plt.colorbar(p,label='dz')
    plt.clim(0, 0.0005)
    plt.plot([0,100],[0,100],color='grey',linestyle='--')
    plt.ylim(0,100)
    plt.xlim(0,100)
    plt.xlabel("Input index")
    plt.ylabel("Predicted index")
    plt.savefig("plots/regress_idx.pdf")
    plt.close()

    plt.hist(dz,bins=25)
    plt.xlabel('dz')
    plt.ylabel('Frequency')
    plt.savefig('plots/dz_hist.pdf')
    plt.close()
    
    plt.hist(target_index_test,label='input',bins=20,alpha=0.7)
    plt.hist(preds_idx,label='predicted',bins=20,alpha=0.7)
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.savefig("plots/idx_hist.pdf")
    plt.close()
    
    return

def create_outlier_plots(preds_z,z_test,preds_idx,target_index_test,absorbers_test):
    #create plots to investigate outliers
    dz = abs(np.array(preds_z) - np.array(z_test))

    outlier_idxs = np.where(dz>0.007)

    for oi in outlier_idxs[0]:
        outlieri = absorbers_test.iloc[oi]
        pred_index = preds_idx[oi]
        input_index = target_index_test[oi]
        plt.step(np.arange(0,100), outlieri.flux)
        plt.vlines(input_index,np.min(outlieri.flux),np.max(outlieri.flux),color = 'g',linestyle='--',label='input index')
        plt.vlines(pred_index,np.min(outlieri.flux),np.max(outlieri.flux),color='r',linestyle='--',label='predicted')
        plt.legend()
        plt.savefig("plots/outlier"+str(oi)+".pdf")
        plt.close()
    
    """
    best_idxs = np.where(dz<0.000015)

    for oi in best_idxs[0]:
        besti = absorbers_test.iloc[oi]
        pred_index = preds_idx[oi]
        input_index = target_index_test[oi]
        plt.step(np.arange(0,100), besti.flux)
        plt.vlines(input_index,np.min(besti.flux),np.max(besti.flux),color = 'g',linestyle='--',label='input index')
        plt.vlines(pred_index,np.min(besti.flux),np.max(besti.flux),color='r',linestyle='--',label='predicted')
        plt.legend()
        plt.savefig("plots/best"+str(oi)+".pdf")
        plt.close()
    """
    return 

def read_data(trainfile,testfile, flag):
    """
    Read in the data files of training and test data from run_RF.py

    Parameters
    ----------
    trainfile: str
        Name of file containing training data used for classifier
    testfile: str
        Name of file containing test data used for classifier
    flag: int   
        Flag for the type of absorber, 1: CIV, 2:MgII

    Returns
    -------
    absorbers: 
        absorber information for all systems with the specified flag
    idx_split: int
        index at which the split between training and test data occurs
    original_id_test: list
        list of indices from the original dataframe
    """

    train_data = pd.read_pickle(trainfile)  
    
    # Include everything but give things which don't match the flag a target 
    # index of 0.
    #absorbers_train = train_data
    # Or just train on CIV?
    absorbers_train = train_data[train_data['isabs'] == flag]

    test_data  = pd.read_pickle(testfile)  
    
    # If running on what we know are absorbers use test_data['isabs'] == flag.
    # If running on things that have been identified by classifier as absorbers
    # use test_data['preds'] == flag
    preds_prob_flag = np.array([i[flag] for i in test_data['preds_probability']])
    absorbers_test = test_data[preds_prob_flag>0.5]
    original_id_test = np.where(preds_prob_flag>0.5)[0]

    #combine so that you can preprocess together
    idx_split = len(absorbers_train)
    absorbers = pd.concat([absorbers_train,absorbers_test])

    return absorbers, idx_split, original_id_test

def run_regressor(X,Y):
    """
    Create the Random Forest regression model and fit the given data

    Parameters
    ----------
    X        
    Y

    Returns
    -------
    model:
        Random forest model fit to the data
    """
    regr = RandomForestRegressor(n_estimators=2000,max_depth=None,
                                      min_samples_split=2,min_samples_leaf=1,max_features="sqrt",
                                      max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                      n_jobs=40,random_state=120,verbose=0)
    model = regr.fit(X,Y)

    return model

def find_target_index(absorbers,zarr, restwl, flag):

    obswl = restwl*(np.array(zarr) + 1)

    target_index = []

    for i in range(len(absorbers)):
        #if (absorbers.iloc[i])['isabs'] != flag:
        #    target_index.append(0)
        #else:
        wi = obswl[i]
        absi = absorbers.iloc[i]
        wavei = absi.wave
        target_index.append(np.where(abs(wavei-wi) == min(abs(wavei - wi)))[0][0])
        
    return target_index

def index_to_redshift(preds_idx,absorbers, restwl):

    preds_wave = []

    for i in range(len(preds_idx)):
        absi = absorbers.iloc[i]
        wavei = absi.wave
        preds_wave.append(wavei[preds_idx[i]])

    preds_z = (np.array(preds_wave)/restwl) -1

    return preds_z

def extractInfo(absorbers):
    z_abs, flux = [], []
    for a in range(len(absorbers)):
        absi = absorbers.iloc[a]
        z_abs.append(absorbers.iloc[a].absInfo[2])    
        flux.append(absi.flux)

    flux=np.array(flux)
    z_abs = np.array(z_abs).astype(float)

    return z_abs,flux

def preprocess(absorbers,target_index,flux,z_abs,idx_split):
    #split into train and test
    absorbers_train = absorbers[:idx_split]
    absorbers_test = absorbers[idx_split:]

    target_index_train = target_index[:idx_split]
    target_index_test = target_index[idx_split:]

    flux_train = flux[:idx_split]
    flux_test = flux[idx_split:]

    z_abs_train = z_abs[:idx_split]
    z_abs_test = z_abs[idx_split:]

    traindict = dict(absorbers=absorbers_train, 
                     target_index=target_index_train,
                     flux = flux_train,
                     z = z_abs_train)

    testdict = dict(absorbers=absorbers_test, 
                    target_index=target_index_test,
                    flux = flux_test,
                    z = z_abs_test)

    return traindict, testdict

def remove_duplicates(testdict,preds_idx,original_id_test):

    test_absorbers = testdict['absorbers']    
    test_specNum = []
    for _,i in enumerate(test_absorbers['absInfo']):
        test_specNum.append(i[3])

    bestfits = []

    for s in np.unique(test_specNum):
        thisspec = np.where(np.array(test_specNum)==s)
        orig_ids_thisspec =  original_id_test[thisspec]

        #find breaks
        breaks = [0]
        for o in range(1,len(orig_ids_thisspec)):
            if orig_ids_thisspec[o] > orig_ids_thisspec[o-1] + 5:
                breaks.append(o)

        #find most central fit
        for b in range(1,len(breaks)):
            distFromCentre = preds_idx[breaks[b-1]:breaks[b]] -50

            # Skip systems which are only identified once or twice as they are more likely to be erroneous
            if len(distFromCentre) >= 3: 
                bestfit = np.where(abs(distFromCentre) == np.min(abs(distFromCentre)))[0][0]
                #print(distFromCentre[bestfit])
                #if abs(distFromCentre[bestfit])>10:
                #    print(s,distFromCentre)
                bestfits.append(orig_ids_thisspec[breaks[b-1] + bestfit])

    chosen_absorbers_list = []
    chosen_preds_idx= []
    chosen_target_idx = []
    chosen_z = []

    target_idx = testdict['target_index']
    test_z = testdict['z']

    for i in range(len(original_id_test)):
        if original_id_test[i] in bestfits:
            dict1 = {}
            dict1.update(test_absorbers.iloc[i])
            chosen_absorbers_list.append(dict1)
            chosen_preds_idx.append(preds_idx[i])
            chosen_target_idx.append(target_idx[i])
            chosen_z.append(test_z[i])

    chosen_absorbers = pd.DataFrame(chosen_absorbers_list)
    chosen_preds_idx = np.array(chosen_preds_idx)
    chosen_target_idx = np.array(chosen_target_idx)
    chosen_z = np.array(chosen_z) 

    return chosen_absorbers, chosen_preds_idx,chosen_target_idx,chosen_z



################################

#for CIV use flag==1, for MgII use flag==2
flag = 1

if flag == 1:
    restwl = 1548.2
elif flag == 2:
    restwl = 2796

print("Reading data...")
absorbers, idx_split,original_id_test = read_data('train_data.pkl','testFine_data.pkl', flag)

print("Extracting absorber information...")
#extract and reformat absorber information
z_abs, flux = extractInfo(absorbers)

print("Finding target index...")
# Identify index in each flux array that is closest to observed wavelength
# This is the "target" for the machine learning
target_index = find_target_index(absorbers,z_abs, restwl, flag)

print("Preprocessing data...")
#preprocess and split into train and test samples
traindict, testdict = preprocess(absorbers,target_index,flux,z_abs,idx_split)

print("Creating and training model...")
#run random forest regression model
model = run_regressor(traindict['flux'], traindict['target_index'])

print("Making predictions...")
#use model to predict index on test sample
preds_idx = model.predict(testdict['flux']).astype(int)

print("Removing duplicates...")
chosen_absorbers, chosen_preds_idx, chosen_target_idx, chosen_z = remove_duplicates(testdict,preds_idx,original_id_test)

#convert index back to redshift
preds_z = index_to_redshift(preds_idx,testdict['absorbers'], restwl)
chosen_preds_z = index_to_redshift(chosen_preds_idx,chosen_absorbers, restwl)

#create plots to see results of regression
#create_regression_plots(preds_z,testdict['z'],preds_idx,testdict['target_index'])
create_regression_plots(chosen_preds_z,chosen_z,chosen_preds_idx,chosen_target_idx)


#create plots to investigate outliers
#create_outlier_plots(preds_z,testdict['z'],preds_idx,testdict['target_index'],testdict['absorbers'])
#create_outlier_plots(preds_z,chosen_z,chosen_preds_idx,chosen_target_idx,chosen_absorbers)





