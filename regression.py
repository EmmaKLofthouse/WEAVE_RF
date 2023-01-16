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

def create_regression_plots(preds_z,civ_z_test,preds_idx,target_index_test):

    dz = abs(np.array(preds_z) - np.array(civ_z_test))

    p = plt.scatter(target_index_test,preds_idx,marker='.',c=dz)
    plt.colorbar(p,label='dz')
    plt.plot([0,100],[0,100],color='grey',linestyle='--')
    plt.ylim(0,100)
    plt.xlim(0,100)
    plt.xlabel("Input index")
    plt.ylabel("Predicted index")
    plt.savefig("regress_idx.pdf")
    plt.close()

    plt.hist(dz,bins=25)
    plt.xlabel('dz')
    plt.ylabel('Frequency')
    plt.savefig('dz_hist.pdf')
    plt.close()
    
    plt.hist(target_index_test,label='input',bins=20,alpha=0.7)
    plt.hist(preds_idx,label='predicted',bins=20,alpha=0.7)
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.savefig("idx_hist.pdf")
    plt.close()
    
    return

def create_outlier_plots(preds_z,civ_z_test,preds_idx,target_index_test):
    #create plots to investigate outliers
    dz = abs(np.array(preds_z) - np.array(civ_z_test))

    outlier_idxs = np.where(dz>0.01)

    for oi in outlier_idxs[0]:
        outlieri = civ_absorbers_test.iloc[oi]
        pred_index = preds_idx[oi]
        input_index = target_index_test[oi]
        plt.step(np.arange(0,100), outlieri.flux)
        plt.vlines(input_index,np.min(outlieri.flux),np.max(outlieri.flux),color = 'g',linestyle='--',label='input index')
        plt.vlines(pred_index,np.min(outlieri.flux),np.max(outlieri.flux),color='r',linestyle='--',label='predicted')
        plt.legend()
        plt.savefig("outlier"+str(oi)+".pdf")
        plt.close()

    best_idxs = np.where(dz<0.000015)

    for oi in best_idxs[0]:
        besti = civ_absorbers_test.iloc[oi]
        pred_index = preds_idx[oi]
        input_index = target_index_test[oi]
        plt.step(np.arange(0,100), besti.flux)
        plt.vlines(input_index,np.min(besti.flux),np.max(besti.flux),color = 'g',linestyle='--',label='input index')
        plt.vlines(pred_index,np.min(besti.flux),np.max(besti.flux),color='r',linestyle='--',label='predicted')
        plt.legend()
        plt.savefig("best"+str(oi)+".pdf")
        plt.close()

    return 

def read_data(trainfile,testfile):
    train_data = pd.read_pickle(trainfile)  
    test_data  = pd.read_pickle(testfile)  

    civ_absorbers_train = train_data[train_data['isabs'] == 1]
    civ_absorbers_test = test_data[test_data['isabs'] == 1]

    mgii_absorbers_train = train_data[train_data['isabs'] == 2]
    mgii_absorbers_test = test_data[test_data['isabs'] == 2]

    #combine so that you can preprocess together
    idx_split = len(civ_absorbers_train)
    civ_absorbers = pd.concat([civ_absorbers_train,civ_absorbers_test])
    mgii_absorbers = pd.concat([mgii_absorbers_train,mgii_absorbers_test])

    return civ_absorbers, mgii_absorbers, idx_split

def run_regressor(X,Y):
    regr = RandomForestRegressor(n_estimators=2000,max_depth=None,
                                      min_samples_split=2,min_samples_leaf=1,max_features="sqrt",
                                      max_leaf_nodes=None,bootstrap=True,oob_score=True,
                                      n_jobs=40,random_state=120,verbose=0)

    model = regr.fit(flux_train, target_index_train)

    return model


################################

civ_absorbers, mgii_absorbers, idx_split = read_data('train_data.pkl','test_data.pkl')

civ_absInfo = civ_absorbers['absInfo']

civ_z = []
for _, val in civ_absInfo.iteritems():
    civ_z.append(val[2])

#convert to observed wavelength
obs_CIV_1548_wave = 1548.2*(np.array(civ_z) + 1)

# Identify index in each flux array that is closest to obs_CIV_1548_wave 
# This is the "target" for the machine learning
target_index = []

for i in range(len(civ_absorbers)):
    wi = obs_CIV_1548_wave[i]
    absi = civ_absorbers.iloc[i]
    wavei = absi.wave
    target_index.append(np.where(abs(wavei-wi) == min(abs(wavei - wi)))[0][0])

flux = []
for _, val in civ_absorbers.flux.iteritems():
    flux.append(val)
flux=np.array(flux)

#split into train and test
flux_train = flux[:idx_split]
flux_test = flux[idx_split:]

target_index_train = target_index[:idx_split]
target_index_test = target_index[idx_split:]

civ_z_test = civ_z[idx_split:]
civ_absorbers_test = civ_absorbers[idx_split:]

#run random forest regression model
model = run_regressor(flux_train, target_index_train)

#use model to predict index on test sample
preds_idx = model.predict(flux_test).astype(int)

#convert index back to redshift
preds_wave = []

for i in range(len(preds_idx)):
    absi = civ_absorbers_test.iloc[i]
    wavei = absi.wave
    preds_wave.append(wavei[preds_idx[i]])

preds_z = (np.array(preds_wave)/1548.2) -1

#create plots to see results of regression
create_regression_plots(preds_z,civ_z_test,preds_idx,target_index_test)

#create plots to investigate outliers
create_outlier_plots(preds_z,civ_z_test,preds_idx,target_index_test)







