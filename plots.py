import numpy as np
import matplotlib.pyplot as plt

def calc_recovery(true, recovered):
    if true == 0:
        return -999,-999
    elif recovered == 0:
        return 0,0
    else:
        frac = recovered/true
        fracerr = frac * np.sqrt((recovered/recovered**2)+(true/true**2))

    return frac, fracerr


def plotRecoveryFraction_type(preds,test_absInfo, sample_size):

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

        print(minN,maxN,recovered_abs_CIV,true_abs_CIV)
        
        #partial absorption
        #true_abs_partialCIV = float(len(preds_bin_partialCIV))
        #recovered_abs_partialCIV = float(len(preds_bin_partialCIV[preds_bin_partialCIV == 3]))
        #true_abs_partialMgII = float(len(preds_bin_partialMgII))
        #recovered_abs_partialMgII = float(len(preds_bin_partialMgII[preds_bin_partialMgII == 3]))

        #calculate recovery fraction
        CIVfrac,CIVfracerr = calc_recovery(true_abs_CIV, recovered_abs_CIV)
        recoveryFracs_CIV.append(CIVfrac)
        recoveryFracsErr_CIV.append(CIVfracerr)

        MgIIfrac,MgIIfracerr = calc_recovery(true_abs_MgII , recovered_abs_MgII)
        recoveryFracs_MgII.append(MgIIfrac)
        recoveryFracsErr_MgII.append(MgIIfracerr)

        #partialfrac,partialfracerr = calc_recovery(true_abs_partialMgII + true_abs_partialCIV, recovered_abs_partialMgII + recovered_abs_partialCIV)
        #recoveryFracs_partial.append(partialfrac)
        #recoveryFracsErr_partial.append(partialfracerr)
    #plt.scatter(logNbins[:-1]+binsize/2,recoveryFracs_MgII,label='MgII')
    #plt.scatter(logNbins[:-1]+binsize/2,recoveryFracs_CIV,label='CIV')
    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_MgII,yerr=recoveryFracsErr_MgII,xerr=binsize/2,linestyle=' ',capsize=3,label='MgII')
    plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_CIV,yerr=recoveryFracsErr_CIV,xerr=binsize/2,linestyle=' ',capsize=3,label='CIV')
    #plt.errorbar(logNbins[:-1]+binsize/2,recoveryFracs_partial,yerr=recoveryFracsErr_partial,xerr=binsize/2,linestyle=' ',capsize=3,label='partial')

    plt.legend()
    plt.ylim(0,1.1)
    plt.title('Identifying correct metal')
    plt.xlabel('logN')
    plt.ylabel('Recovery Fraction')
    plt.savefig('plots/rf_spec' + str(sample_size) + '.pdf')
    plt.close()

    return
    
    
def plotIdentifications(test_isabs,preds,test_absInfo, sample_size):
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
    plt.savefig('plots/idents_CIV_spec' + str(sample_size) +'.pdf')
    plt.close()

    #Plot MgII results
    plt.fill_between(logNbins[:-1]+binsize/2,Total_MgII,np.array(Correct_MgII) + np.array(MisIdentified_MgII),color='k',label='All MgII', step="pre", alpha=0.2)
    plt.fill_between(logNbins[:-1]+binsize/2,Correct_MgII,color='g',label='Correctly identified as MgII', step="pre", alpha=0.4)
    plt.fill_between(logNbins[:-1]+binsize/2,np.array(Correct_MgII) + np.array(MisIdentified_MgII),Correct_MgII,color='r',label='Plus Incorrectly identified as CIV', step="pre", alpha=0.4)

    plt.legend()
    plt.xlabel('logN')
    plt.ylim(0,np.max(Total_MgII)*1.2)

    plt.ylabel('Number of absorbers')
    plt.savefig('plots/idents_MgII_spec' + str(sample_size) +'.pdf')
    plt.close()

    return

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


