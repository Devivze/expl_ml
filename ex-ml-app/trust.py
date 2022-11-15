import numpy as np
import pandas as pd
from retroviz import RetroScore, run_retro_score
import pickle

#%%Unpacking
with open('model_set.pkl', 'rb') as f:
    model_set = pickle.load(f)
    
keys = list(model_set.keys())
for i in keys:
    locals()[i] = model_set[i]

del(model_set,keys,i,f)


#%% Rertro trust
X_train_tr = X_train.values
X_valid_tr = X_valid.values
y_valid_tr = y_valid.values
y_train_tr = y_train.values
del(X_train,X_valid,y_valid,y_train)

rs = RetroScore(k=5)
pred = model.predict(X_valid_tr)
train_pred = model.predict(X_train_tr)
retro_score, unnormalized_score, nbs_x, nbs_y = run_retro_score(rs, X_train_tr, y_train_tr, X_valid_tr, pred, train_pred)

for k in range(0,len(retro_score)-1):
    if retro_score[k] != 0 and retro_score[k] != 1:
        retro_score[k] = retro_score[k][0]
#%%Find bins for plot
error = abs(y_valid_tr.reshape(-1, 1)-pred.reshape(-1,1))

bins = np.histogram(error, bins=10)[1].astype('float')
binned = np.digitize(error, bins)


binlabel = np.zeros((binned.shape[0])).astype('float')
for ix in range(binlabel.shape[0]):
    binlabel[ix] = round(bins[binned[ix]-1][0], 3)

# find average error in each bin
df = pd.DataFrame({"normalized_score": retro_score.astype('float').flatten(),"bin": binlabel.flatten(), "error": error.flatten()})
mean = df.groupby("bin").mean()["normalized_score"].reset_index()
std = df.groupby("bin").std()["normalized_score"].reset_index().fillna(1e-6)

   
#%%plot normalization
y_max = mean.normalized_score-std.normalized_score
y_max = y_max.rename("y_max")
y_min = mean.normalized_score+std.normalized_score
y_min = y_min.rename("y_min")
score = mean.normalized_score
score = score.rename("score")
score_df = pd.concat([mean.bin,y_max,y_min,score], axis=1)

#%% Save
trust_set = {'score': score_df,
            'retro_score': retro_score,
            'nbs_x': nbs_x,
            'nbs_y': nbs_y}

with open('trust_set.pkl','wb') as f:
    pickle.dump(trust_set,f) 