import pandas as pd

def saveTrainData(train, train_isabs, train_absInfo, train_vel, train_wave, filename):
    d = {'flux': train,
         'isabs': train_isabs, 
         'absInfo': train_absInfo, 
         'vel': train_vel,
         'wave': train_wave}

    df = pd.DataFrame(data=d)
    df.to_pickle(filename)

    return

def saveTestData(test, test_isabs, test_absInfo, test_vel, test_wave, preds,
                 preds_probability, filename):
    
    d = {'flux': test,
         'isabs': test_isabs, 
         'absInfo': test_absInfo, 
         'vel': test_vel,
         'wave':test_wave,   
         'preds':preds,
         'preds_probability': list(preds_probability)}

    df = pd.DataFrame(data=d)
    df.to_pickle(filename)

    return
