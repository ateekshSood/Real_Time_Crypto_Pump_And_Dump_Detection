# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: real-time-crypto-pump-and-dump-detection
#     language: python
#     name: python3
# ---

# %%
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("data/processed/processed_data.csv")

#fetch the clean csv file

# %%
df.head()

# %%
df.info()

# %%
df['Open Time'] = pd.to_datetime(df['Open Time'])

# %%
df.set_index('Open Time' , inplace=True)
print(df.index.name)

# %%
df['Hour_of_day'] = df.index.hour
df['Day_of_week'] = df.index.day_of_week

# %%
df.info()

# %%
df.columns

# %%
from sklearn.model_selection import train_test_split

y = df['Target_Pump']
x = df.drop(columns=['Target_Pump' , 'Open Price' , 'Close Price'])
x_shift = x.shift(5)
x_shift = x_shift.dropna()
y_shift = y.iloc[5:]


# %%
total_rows = len(df)
train_end = int(total_rows * 0.80)
val_end = int(total_rows * 0.90)

# 80% train , 10 % val and 10% test

x_train , y_train = x_shift.iloc[:train_end] , y_shift.iloc[:train_end]
x_val , y_val = x_shift.iloc[train_end:val_end] , y_shift.iloc[train_end:val_end]
x_test , y_test = x_shift.iloc[val_end:] , y_shift.iloc[val_end:]

# %%
# from pandas.plotting import scatter_matrix

# attributes = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume',
#        'Change_1min', 'Change_3min', 'Change_5min', 'Vol_avg_15min',
#        'Volume_spike_ratio', 'High_Low_spread', 'Relative_strength_index',
#        'Target_Pump', 'Hour_of_day', 'Day_of_week']

# scatter_matrix(df[attributes])
# plt.tight_layout()
# plt.show()




# %%
(y_train == 1).sum()

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV  
from sklearn.model_selection import TimeSeriesSplit

tree = DecisionTreeClassifier(random_state=42 , class_weight='balanced')

param_list = {
    'max_depth' : [5 , 10 , 20],
    'min_samples_split' : [2 , 10 , 50],
    'min_samples_leaf' : [1,5,10]
}

random_search = RandomizedSearchCV(
    estimator=tree,
    param_distributions=param_list,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    cv=TimeSeriesSplit(n_splits=5)
)

random_search.fit(x_train , y_train )


# %%
random_search.best_params_

# %%
random_search.best_score_

# %%
tree_best = random_search.best_estimator_
tree_best.fit(x_train , y_train)


# %%
from sklearn.metrics import recall_score

y_pred = tree_best.predict(x_val)
print(f"Recall score : {recall_score(y_val , y_pred):.3f}" )

# %%
from sklearn.metrics import ConfusionMatrixDisplay , confusion_matrix

conf_mat = confusion_matrix(y_val , y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
disp.plot()
plt.show()

# %%
from sklearn.metrics import precision_score

print(f"Precision : {precision_score(y_val , y_pred)}")

# %%

feat_importances = pd.Series(tree_best.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("What is the Tree actually looking at")
plt.show()

# %%
from xgboost import XGBClassifier

# %%
ratio = (y_train == 0).sum()/(y_train == 1).sum()

sqrt_ratio = np.sqrt(ratio)
xg_boost = XGBClassifier(scale_pos_weight=ratio ,
                         random_state = 42,
                         verbosity=2)

xg_param_list = {
    # 1. Tree Architecture (Keep them shallow to avoid memorization)
    'max_depth': [3, 4, 5, 6], 
    
    # 2. Learning Speed (Slower learn, more trees = more stable)
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    
    # 3. Bagging / Randomization (Forces model to generalize)
    'subsample': [0.6, 0.8, 1.0],           # Don't look at all rows for every tree
    'colsample_bytree': [0.6, 0.8, 1.0],    # Don't look at all columns for every tree
    
    # 4. Pruning and Regularization (The ultimate overfitting killers)
    'gamma': [0, 0.1, 0.5, 1],              # Minimum loss reduction needed to split
    'reg_alpha': [0, 0.1, 1],               # L1 Heavy penalty
    'reg_lambda': [1, 10, 100]              # L2 Heavy penalty
}


cross_val_xg = RandomizedSearchCV(xg_boost , param_distributions= xg_param_list , n_jobs=-1,
                                  cv= TimeSeriesSplit(n_splits=5) , scoring='f1')

cross_val_xg.fit(x_train , y_train)



# %%
xg_best = cross_val_xg.best_estimator_
xg_best.fit(x_train , y_train)

y_pred_xg = xg_best.predict(x_val)

print(f"recall : {recall_score(y_val , y_pred_xg)}")
print(f"precision : {precision_score(y_val , y_pred_xg)}")

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
log_reg_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42,verbose=2 , class_weight='balanced')
    ) 

params = [
    {
        # liblinear only supports l1 and l2, and ignores l1_ratio
        "logisticregression__solver": ["liblinear"],
        "logisticregression__penalty": ["l1", "l2"],
        "logisticregression__C": [0.01, 0.1, 1, 10, 100],            
        "logisticregression__max_iter": [1000, 2500] 
    },
    {
        # saga supports elasticnet, but requires l1_ratio
        "logisticregression__solver": ["saga"],
        "logisticregression__penalty": ["elasticnet"],
        "logisticregression__l1_ratio": [0.2, 0.5, 0.8], # Try different mix ratios
        "logisticregression__C": [0.01, 0.1, 1, 10, 100],
        "logisticregression__max_iter": [1000, 2500]
    }
]

cross_vaL_log_reg = RandomizedSearchCV(log_reg_pipeline ,param_distributions=params,random_state=42,
                   verbose=2 , cv=TimeSeriesSplit(n_splits=5) , n_jobs=-1 , scoring='f1')

cross_vaL_log_reg.fit(x_train , y_train)


# %%
log_reg_best = cross_vaL_log_reg.best_estimator_

log_reg_best.fit(x_train , y_train)


# %%
y_pred_log_reg = log_reg_best.predict(x_val)

print(f"recall {recall_score(y_val , y_pred_log_reg)}")
print(f"precision {precision_score(y_val , y_pred_log_reg)}")

# %%
from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(is_unbalance=True)

lgbm_param_list = {

    'max_depth': [3, 5, 7, -1],         # -1 means no limit, allowing num_leaves to control it
    'num_leaves': [15, 31],         # Keep smaller to prevent overfitting (rule: should be < 2^max_depth)
    
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    
    'subsample': [0.6, 0.8, 1.0],           # Row sampling
    'colsample_bytree': [0.6, 0.8, 1.0],    # Feature sampling
    
    'min_child_samples': [10, 20, 50],      # Minimum data points required in a leaf
    'reg_alpha': [0, 0.1, 1, 10],           # L1 penalty
    'reg_lambda': [0.1, 1, 10, 100]         # L2 penalty
}

cross_val_lgb = RandomizedSearchCV(lgbm_clf , param_distributions= lgbm_param_list , scoring="precision" ,
                                   random_state=42 , cv=TimeSeriesSplit(n_splits=5))
cross_val_lgb.fit(x_train , y_train)


# %%
light_gbm_best = cross_val_lgb.best_estimator_
light_gbm_best.fit(x_train , y_train)

y_pred_lgb = light_gbm_best.predict(x_val)

print(f"recall : {recall_score(y_val , y_pred_lgb)}")
print(f"precision : {precision_score(y_val , y_pred_lgb)}")


# %%
from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(random_state=42 , auto_class_weights='Balanced' , verbose=0)


cat_param_list = {
    # 1. Tree Architecture
    'depth': [3, 4, 5, 6],                  # CatBoost prefers shallow, symmetric trees (Keep <= 6 for speed)
    
    # 2. Learning Speed
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300],          # CatBoost's name for n_estimators
    
    # 3. Regularization
    'l2_leaf_reg': [1, 3, 5, 10, 50],       # L2 penalty coefficient (higher = more conservative)
    
    # 4. Randomization
    'subsample': [0.6, 0.8, 1.0],           # Only works if bootstrap_type is 'MVS' or 'Poisson' (Default is usually fine)
    'random_strength': [0.1, 1, 10]         # Gives random variance to splits to prevent overfitting
}


cat_clf_cross_val = RandomizedSearchCV(estimator=cat_clf , param_distributions= cat_param_list , scoring='f1' ,
                                       random_state=42 , cv=TimeSeriesSplit(n_splits=5))
cat_clf_cross_val.fit(x_train , y_train)

# %%
best_cat_boost = cat_clf_cross_val.best_estimator_
best_cat_boost.fit(x_train , y_train )
y_pred_cat = best_cat_boost.predict(x_val)

print(f"recall : {recall_score(y_val , y_pred_cat)}")
print(f"precision : {precision_score(y_val , y_pred_cat)}")

# %%
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

estimators_list = [("best_treee" , tree_best) , 
                   ("best_xg" , xg_best), 
                   ("best_log_reg" , log_reg_best) , 
                   ("best_lgbm" , light_gbm_best) , 
                   ("best_cat" , best_cat_boost)]

voting_clf = VotingClassifier(estimators=estimators_list , voting='soft')

cross_val_voting = cross_val_score(estimator=voting_clf , cv=TimeSeriesSplit(n_splits=5) , X=x_train , y=y_train , scoring='f1')
cross_val_voting

# %%
voting_clf.fit(x_train , y_train)

y_pred_final = voting_clf.predict(x_val)

print(f"recall : {recall_score(y_val , y_pred_final)}")
print(f"precision : {precision_score(y_val , y_pred_final)}")


# %%
\

