import pandas as pd
import shap
import pickle
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp
import matplotlib.pyplot as plt
#%% Model unpacking
with open('model_set.pkl', 'rb') as f:
    model_set = pickle.load(f)
    
keys = list(model_set.keys())
for i in keys:
    locals()[i] = model_set[i]

del(model_set,keys,i,f)
features = list(X_train.columns)
#%% Permutation Importance

perm = PermutationImportance(model).fit(X_valid, y_valid)
Permutation_Importance = eli5.explain_weights_df(perm, feature_names = features)

#%% Shap
explainer = shap.TreeExplainer(model)

#Summary plot
shap_values = explainer.shap_values(X_valid.iloc[0:5000])
shap.initjs()
fig = shap.summary_plot(shap_values, X_valid.iloc[0:5000], show = False)
plt.savefig('D:/Home/.MT/dashapp/App/assets/shap.png',bbox_inches='tight')

#Single prediction
data = X_valid.iloc[1000:1001]
shap_values_single = explainer.shap_values(data)
prediction = model.predict(data)
base = explainer.expected_value

SHAP = {'shap_values': shap_values,
            'shap_values_single': shap_values_single,
            'base': base,
            'prediction': prediction,
            'explainer': explainer}

#%% PDP
PDP_isolate = {}
PDP_interact = {}
features_list = features[:]
#Isolate
for ft in features:
    pdp_dist = pdp.pdp_isolate(model=model,
                           dataset=X_valid,
                           model_features=features,
                           feature=ft,
                           num_grid_points = 14,
                           grid_type='equal')
    grid = pdp_dist.feature_grids
    dist = pdp_dist.pdp - min(pdp_dist.pdp)
    s = pd.Series({'grid': grid, 'dist': dist})
    PDP_isolate[ft] = s
    
#Interact
for ft1 in features:
    features_list.remove(ft1)
    for ft2 in features_list:
            PDP_val = pdp.pdp_interact(model=model,
                                            dataset=X_valid,
                                            model_features=features,
                                            features=[ft1, ft2],
                                            # num_grid_points = [15,15]
                                            )

            key = [ft1 + ft2]
            PDP_interact[str(key)]= PDP_val


#%% Save

Expl_set = {'Permutation_Importance': Permutation_Importance,
            'SHAP': SHAP,
            'PDP_isolate': PDP_isolate,
            'PDP_interact': PDP_interact}

with open('Expl_set.pkl','wb') as f:
    pickle.dump(Expl_set,f) 




    