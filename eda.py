import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def get_scaled_imgs(df):
    imgs = []
    label =[]

    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        imgs.append(np.dstack((a, b, c)))
    return np.array(imgs)
                    

def get_data():
    train = pd.read_json('/kaggle/working/data/processed/train.json')
    test = pd.read_json('/kaggle/working/data/processed/test.json')
    scale_train = get_scaled_imgs(train)
    scale_test = get_scaled_imgs(test)
    target = np.array(scale_train['is_iceberg'])
    x_train,x_validation,y_train,y_validation=train_test_split(scale_train,
                                                   target,
                                                   test_size=0.3,
                                                   random_state=1,
                                                   stratify=target)
    
    return x_train, x_validation, y_train, y_validation, scale_test