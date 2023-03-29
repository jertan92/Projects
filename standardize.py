import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
def standard(data):
    X_train_dep_2 = pd.read_csv('../capstone/X_train_dep_2.csv')
    ss = StandardScaler()
    X_train_dep_2_sc = ss.fit_transform(X_train_dep_2)
    std_data = ss.transform(data)
    return std_data
