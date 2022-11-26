import pickle
import os 
import pandas as pd
import numpy as np
import lightgbm


# with open("./ckpt/lightgbm_ckpt_split_10_0408.pkl","rb") as f:
#     model = pickle.load(f)
# f.close()
models = []
for idx in range(5):
    with open(f"./ckpt/lightgbm_0408_ckpt_split_k_{idx}.pkl","rb") as f:
        model = pickle.load(f)
        models.append(model)
    f.close()
with open("./feat_list/lightgbm_feats.pkl","rb") as g:
    feature_ls = pickle.load(g)
g.close()
feature_ls += ['investment_id']

import ubiquant
env = ubiquant.make_env()
iter_test = env.iter_test()

def prepare_test_data(df):
    operation_ls = ["sum","mean","min","max","median","std","var"]
    ori_feat_size = len(df.columns) - 3
    add_feat_df = None
    for opt in operation_ls:
        print(f"Begin process operation:{opt}")
        prefix = opt+"_"
        # prefix_time = prefix+"time_id"
        a = df.groupby("investment_id").transform(opt).add_prefix(prefix)
        # a.drop([prefix_time],axis=1,inplace=True)
        add_feat_df = pd.concat([a,add_feat_df],axis=1)
    
    df = pd.concat([df,add_feat_df],axis=1)
    # drop features
    good_feats = list(df.columns)
    rm_feats = ['investment_id','target'] # no time_id
    good_feats = [f for f in good_feats if f not in rm_feats]
    good_feats = [f for f in good_feats if not f.endswith("target")]
    return df, good_feats

def infer(models):
    preds = []
    for model in models:
        pred = model.predict(test_df_aug[good_feats])
        preds.append(pred)
    return np.mean(preds, axis=0)

for (test_df, sample_prediction_df) in iter_test:
    test_df = test_df[feature_ls]
    test_df_aug, good_feats = prepare_test_data(test_df)
    # test_df.drop(['row_id','investment_id'], axis=1, inplace=True)
    # pred = model.predict(test_df_aug[good_feats])
    pred = infer(models)
    sample_prediction_df['target'] = list(pred)
    env.predict(sample_prediction_df) 