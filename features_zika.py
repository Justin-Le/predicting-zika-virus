import numpy as np
import pandas as pd

def extract_features(data, target):
    ######################################## 
    # Create binary features
    ######################################## 

    # -100 < X < -70
    # 20 < Y < 40
    X_neg100_neg70 = np.where(data['X'] > -100, 1, 0) -\
                     np.where(data['X'] > -70, 1, 0)

    Y_20_40 = np.where(data['Y'] > 20, 1, 0) -\
              np.where(data['Y'] > 40, 1, 0)

    data['X_neg100_neg70_Y_20_40'] = X_neg100_neg70*Y_20_40

    # -110 < X < -50
    # -40 < Y < 30
    X_neg110_neg50 = np.where(data['X'] > -110, 1, 0) -\
                     np.where(data['X'] > -50, 1, 0)

    Y_neg40_30 = np.where(data['Y'] > -40, 1, 0) -\
                 np.where(data['Y'] > 30, 1, 0)

    data['X_neg110_neg50_Y_neg40_30']= X_neg110_neg50*Y_neg40_30

    """
    # GAUL_AD0 == 64 encoded (COUNTRY == Taiwan)
    data['country_is_taiwan'] = np.where(data['GAUL_AD0'] == 64, 1, 0)

    # GAUL_AD0 == 56 encoded (COUNTRY == Brazil)
    data['country_is_brazil'] = np.where(data['GAUL_AD0'] == 56, 1, 0)
    """

    return data, target

if __name__ == "__main__":
    extract_features()
