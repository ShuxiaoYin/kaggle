import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import joblib
import pickle

sample = True
n_features = 300
features = [f'f_{i}' for i in range(n_features)]
save_fig_dir = "/share/zongyu/task_embedding/ours/torch_ubi/figure"
save_model_dir = "/share/zongyu/task_embedding/ours/torch_ubi/ckpt"
train = pd.read_pickle('/share/zongyu/task_embedding/ours/torch_ubi/data/ubi/train.pkl')
if sample:
    train = train.sample(frac=0.01,random_state=42)
train.index = list(range(len(train)))
investment_id = train.pop("investment_id")

_ = train.pop("row_id")
_ = train.pop("time_id")
y = train.pop("target")

investment_ids = list(investment_id.unique())
investment_id_size = len(investment_ids) + 1
# test_file_save = open("/share/zongyu/task_embedding/ours/torch_ubi/feat_list/a.pkl","wb")
# test_file_read = open("/share/zongyu/task_embedding/ours/torch_ubi/feat_list/a.pkl","rb")
# with open("/share/zongyu/task_embedding/ours/torch_ubi/feat_list/investment_id_dnn.pkl","wb") as f:
#     pickle.dump(investment_ids,f)
# print("finish saving!")
investment_id_lookup_layer = layers.IntegerLookup(max_tokens=investment_id_size)
investment_id_lookup_layer.adapt(pd.DataFrame({"investment_ids":investment_ids}))

def preprocess(X, y):
    return X, y
def make_dataset(feature, investment_id, y, batch_size=1024, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature), y))
    ds = ds.map(preprocess)
    if mode == "train":
        ds = ds.shuffle(4096)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_model():
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = layers.Embedding(investment_id_size, 32, input_length=1)(investment_id_x)
    investment_id_x = layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    
    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    
    x = layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = layers.Dense(512, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    return model


model = get_model()
model.summary()
# keras.utils.plot_model(model, show_shapes=True)


kfold = StratifiedKFold(5, shuffle=True, random_state=42)
models = []
best_mse = []
for index, (train_indices, valid_indices) in enumerate(kfold.split(train,investment_id)):
    X_train, X_val = train.iloc[train_indices], train.iloc[valid_indices]
    investment_id_train = investment_id[train_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[valid_indices]
    investment_id_val = investment_id[valid_indices]
    train_ds = make_dataset(X_train, investment_id_train, y_train)
    valid_ds = make_dataset(X_val, investment_id_val, y_val, mode="valid")
    model = get_model()
    checkpoint = keras.callbacks.ModelCheckpoint(f"model_0329_{index}.tf",monitor="val_rmse",save_best_only=True, save_weights_only=True)
    early_stop = keras.callbacks.EarlyStopping(patience=10)
    history = model.fit(train_ds, epochs=30, validation_data=valid_ds, callbacks=[checkpoint, early_stop])
    models.append(keras.models.load_model(f"model_0329_{index}"))
    pearson_score = stats.pearsonr(model.predict(valid_ds).ravel(), y_val.values)[0]
    print('Pearson:', pearson_score)
    pd.DataFrame(history.history, columns=["mse", "val_mse"]).plot()
    plt.title("MSE")
    plt.savefig(os.path.join(save_fig_dir,"MSE.png"))
    plt.show()
    pd.DataFrame(history.history, columns=["mae", "val_mae"]).plot()
    plt.title("MAE")
    plt.savefig(os.path.join(save_fig_dir,"MAE.png"))
    plt.show()
    pd.DataFrame(history.history, columns=["rmse", "val_rmse"]).plot()
    plt.title("RMSE")
    plt.savefig(os.path.join(save_fig_dir,"RMSE.png"))
    plt.show()

    #
    best_mse.append(max(history.history['val_mse']))

    del investment_id_train
    del investment_id_val
    del X_train
    del X_val
    del y_train
    del y_val
    del train_ds
    del valid_ds
    gc.collect() # clean om
    break


# select best model
idx_ls = list(range(len(best_mse)))
a = sorted(list(zip(idx_ls,best_mse)),key=lambda x:x[1])
best_id = a[0][0] # choosing the model with the least MSE
best_model = models[best_id]
print("finish training...")
# save model
# model_path = os.path.join(save_model_dir,"dnn_ensemble_best_0329.ckpt")
# best_model.save_weights(model_path)
# model_path = os.path.join(save_model_dir,"dnn_ensemble_best_0329.pkl")
# joblib.dump(best_model,model_path)
