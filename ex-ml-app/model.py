import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pickle
import numpy as np

column_names = ['Cell ID', 'sim_time', 'centre SINR dBm', 'edge SINR dBm', 'centre RSRP dBm', 'edge RSRP dBm',
'interference_level1', 'interference_level2', 'interference_level3', 'interference_level4',
'Intial_Number_Of_UEs', 'percentageCentreUsers',' 1 ', 'PCF', 'ECB',
'Antenna Downtilt', '2', 'cell RSRP', 'Average Interference Created','3' , 'current_centre_SINR_dB',
'current_edge_SINR_dB', 'current_centre_RSRP_dB', 'current_edge_RSRP_dB', 'current_interference_level1',
'current_interference_level2', 'current_interference_level3', 'current_interference_level4',
'current_Intial_Number_Of_UEs', 'current_percentageCentreUsers']


df = pd.read_csv('Temp_ICIC_CCO_SON_Stats.txt', sep='\t', names=column_names, index_col=False)
df = df.loc[100000:150000,:]

# Select subset of predictors
cols_to_use =  ['PCF', 'ECB', 'Antenna Downtilt','centre SINR dBm', 'edge SINR dBm', 'centre RSRP dBm', 'edge RSRP dBm']

#%% Select target
scaler = MinMaxScaler()

for col in cols_to_use:
    
    tmp_plus = df[col].mean() + 3*df[col].std()
    tmp_minus =  df[col].mean() - 3*df[col].std()
    df = df.loc[df[col] < tmp_plus]
    df = df.loc[df[col] > tmp_minus]
    # df[col].hist()
    

mon_cols = ['Cell ID','Average Interference Created','cell RSRP','sim_time']
monitoring = df[mon_cols]

df['cell RSRP'] = scaler.fit_transform(np.array([df['cell RSRP']]).T)
df['Average Interference Created'] = scaler.fit_transform(np.array([df['Average Interference Created']]).T)

#%%Calculating GKPI

alpha = 0.6
KPI = (alpha*df['cell RSRP'])-((1-alpha)*df['Average Interference Created'])
a = (KPI - KPI.min())
KPI = a/ a.max()

df.loc[:,'KPI'] = KPI
monitoring.loc[:,'KPI'] = KPI
y = KPI 
X = df[cols_to_use]
#%%
# for col in cols_to_use:
#  a = min(df[col])
#  print (col,a)

#%% Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
#Fit the model
ICIC_CCO_model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=4)
ICIC_CCO_model.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_valid, y_valid)],verbose=False)

model_set = {'model': ICIC_CCO_model,
            'X_train': X_train,
            'X_valid': X_valid,
            'y_valid': y_valid,
            'y_train': y_train,
            'monitoring' :monitoring}


with open('model_set.pkl','wb') as f:
    pickle.dump(model_set,f)
























