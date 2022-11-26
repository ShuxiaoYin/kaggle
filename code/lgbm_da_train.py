# add data augmentation

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.python.ops import math_ops


def correlationMetric(x, y, axis=0):
    """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
    x = tf.convert_to_tensor(x)
    y = math_ops.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    # return tf.constant(1.0, dtype=x.dtype) - corr
    return corr

def p_score(y_true, y_pred):
    eval_name = "pearson_score"
    is_higher_better = True
    eval_result = correlationMetric(y_true, y_pred)
    return eval_name, eval_result, is_higher_better

date = "0409"
data_types_dict = {
    'time_id': 'int32',
    'investment_id': 'int16',
    "target": 'float16',
}

features = [f'f_{i}' for i in range(300)]

for f in features:
    data_types_dict[f] = 'float16'
    
target = 'target'

data_path = "/share/zongyu/task_embedding/ours/torch_ubi/data/ubi/train.csv"
# data_path = "/home/yangzl/data/ubi/train.csv"
train_df = pd.read_pickle('/share/zongyu/task_embedding/ours/torch_ubi/data/ubi/train.pkl')
# train_df = train_df.sample(frac=0.1,random_state=42)
# load feature list
feats_ls = pickle.load(open("./feat_list/lightgbm_feats.pkl","rb"))
feats_ls += ["time_id","investment_id","target"]

train_df = train_df[feats_ls]
# train_df = train_df[feats_ls].sample(frac=0.01)

operation_ls = ["sum","mean","min","max","median","std","var"]
operation_lss = operation_ls[:2]
ori_feat_size = len(train_df.columns) - 3
print(f"Now we have {ori_feat_size} features in total.")
add_feat_df = None
for opt in operation_ls:
    print(f"Begin process operation:{opt}")
    prefix = opt+"_"
    prefix_time = prefix+"time_id"
    a = train_df.groupby("investment_id").transform(opt).add_prefix(prefix)
    a.drop([prefix_time],axis=1,inplace=True)
    add_feat_df = pd.concat([a,add_feat_df],axis=1)
    print(f"Now we add {len(a.columns)} features in total.")

train_df = pd.concat([train_df,add_feat_df],axis=1)
cur_feat_size = len(train_df.columns)-3 # time_id, investment_id, target
print(f"Now we have {cur_feat_size} features in total.")

good_feats = list(train_df.columns)
rm_feats = ['investment_id','time_id','target']
good_feats = [f for f in good_feats if f not in rm_feats]
good_feats = [f for f in good_feats if not f.endswith("target")]


seed = 0
folds = 5
models = []

skf = StratifiedKFold(folds, shuffle = True, random_state = seed)

print("Begin training...")
for fold, (train_index, test_index) in enumerate(skf.split(train_df, train_df['investment_id'])):
    train = train_df.iloc[train_index]
    valid = train_df.iloc[test_index]
    
    lgbm = LGBMRegressor(
        num_leaves=2 ** np.random.randint(3, 8),
        learning_rate = 10 ** (-np.random.uniform(0.1,2)),
        n_estimators = 1000,
        min_child_samples = 1000, 
        subsample=np.random.uniform(0.5,1.0), 
        subsample_freq=1,
        n_jobs= -1
    )
    lgbm.fit(train[good_feats], train[target], 
            eval_set = (valid[good_feats], valid[target]),
            early_stopping_rounds = 10,
            # eval_metric = p_score,
            )
    models.append(lgbm)
    with open(f"/share/zongyu/task_embedding/ours/torch_ubi/ckpt/lightgbm_{date}_ckpt_split_k_{fold}.pkl","wb") as f:
        pickle.dump(lgbm, f)
    f.close()

# with open(f"/share/zongyu/task_embedding/ours/torch_ubi/ckpt/lightgbm_ckpt_split_10_{date}.pkl","wb") as f:
#     pickle.dump(lgbm, f)
# print("finish save")



# import lightgbm
# lightgbm.plot_importance(lgbm, figsize = (20, 60))

# save importance
# booster = lgbm.booster_
# importance = booster.feature_importance(importance_type='split')
# feature_name = booster.feature_name()
# # for (feature_name,importance) in zip(feature_name,importance):
# #     print (feature_name,importance) 
# feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
# feature_importance.to_csv('feature_importance.csv',index=False)
