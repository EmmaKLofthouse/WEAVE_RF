"""
Run random forest regression to identify redshift of absorbers

"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

def remove_duplicates():
    """
    remove absorbers which are in the same spectrum, at roughly the same redshift
    !!! Need to do the regression first to identify redshift as this won't be 
    known a priori in the real data set
    """
    #loop over each entry, check if it's been seen before and if not add to output

    return

def create_regression_plots(preds_z,z_test,preds_idx,target_index_test):

    dz = abs(np.array(preds_z) - np.array(z_test))

    p = plt.scatter(target_index_test,preds_idx,marker='.',c=dz)
    plt.colorbar(p,label='dz')
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
    train_data = pd.read_pickle(trainfile)  
    test_data  = pd.read_pickle(testfile)  

    absorbers_train = train_data[train_data['isabs'] == flag]
    absorbers_test = test_data[test_data['isabs'] == flag]

    #combine so that you can preprocess together
    idx_split = len(absorbers_train)
    absorbers = pd.concat([absorbers_train,absorbers_test])

    return absorbers, idx_split

def run_regressor(X,Y):
    regr = RandomForestRegressor(n_estimators=2000,max_depth=None,
                                      min_samples_split=2,min_samples_leaf=1,max_features="sqrt",
                                      max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                      n_jobs=40,random_state=120,verbose=0)

    model = regr.fit(X,Y)

    return model

def find_target_index(absorbers,zarr, restwl):

    obswl = restwl*(np.array(zarr) + 1)

    target_index = []

    for i in range(len(absorbers)):
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

################################

#for CIV use flag==1, for MgII use flag==2
flag = 1

if flag == 1:
    restwl = 1548.2
elif flag == 2:
    restwl = 2796

absorbers, idx_split = read_data('train_data.pkl','test_data.pkl', flag)

#extract and reformat absorber information
z_abs, flux = extractInfo(absorbers)

# Identify index in each flux array that is closest to observed wavelength
# This is the "target" for the machine learning
target_index = find_target_index(absorbers,z_abs, restwl)

#preprocess and split into train and test samples
traindict, testdict = preprocess(absorbers,target_index,flux,z_abs,idx_split)

#run random forest regression model
model = run_regressor(traindict['flux'], traindict['target_index'])

#use model to predict index on test sample
preds_idx = model.predict(testdict['flux']).astype(int)

#convert index back to redshift
preds_z = index_to_redshift(preds_idx,testdict['absorbers'], restwl)

#create plots to see results of regression
create_regression_plots(preds_z,testdict['z'],preds_idx,testdict['target_index'])

#create plots to investigate outliers
create_outlier_plots(preds_z,testdict['z'],preds_idx,testdict['target_index'],testdict['absorbers'])







