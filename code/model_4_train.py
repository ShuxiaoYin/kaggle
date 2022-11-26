import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pickle as pkl
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import backend as K
from keras import regularizers 
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from sklearn.base import clone
from typing import Dict
import matplotlib.pyplot as plt
from scipy import stats
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold, GroupKFold
from tqdm import tqdm
from tensorflow.python.ops import math_ops
sample = False
n_features = 300
features = [f'f_{i}' for i in range(n_features)]
feature_columns = ['investment_id', 'time_id'] + features
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
def make_dataset_2(feature, investment_id, y, batch_size=1024, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(((feature), y))
    ds = ds.map(preprocess)
    if mode == "train":
        ds = ds.shuffle(4096)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds
def make_test_dataset2(feature, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(((feature)))
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
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


def get_model2():
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = layers.Embedding(investment_id_size, 32, input_length=1)(investment_id_x)
    investment_id_x = layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)    
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
   # investment_id_x = layers.Dropout(0.65)(investment_id_x)
   
    
    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dropout(0.65)(feature_x)
    
    x = layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = layers.Dense(512, activation='swish', kernel_regularizer="l2")(x)
   # x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer="l2")(x)
  #  x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.75)(x)
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    return model


def get_model3():
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float32)
    
    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = layers.Embedding(investment_id_size, 32, input_length=1)(investment_id_x)
    investment_id_x = layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dropout(0.5)(investment_id_x)
    investment_id_x = layers.Dense(32, activation='swish')(investment_id_x)
    investment_id_x = layers.Dropout(0.5)(investment_id_x)
    #investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    
    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dropout(0.5)(feature_x)
    feature_x = layers.Dense(128, activation='swish')(feature_x)
    feature_x = layers.Dropout(0.5)(feature_x)
    feature_x = layers.Dense(64, activation='swish')(feature_x)
    
    x = layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1)(x)
    output = tf.keras.layers.BatchNormalization(axis=1)(output)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    return model
def correlationMetric(x, y, axis=-2):
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
    return tf.constant(1.0, dtype=x.dtype) - corr


def correlationLoss(x,y, axis=-2):
    """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
    while trying to have the same mean and variance"""
    x = tf.convert_to_tensor(x)
    y = math_ops.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    xsqsum = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
    ysqsum = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xsqsum * ysqsum)
    return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr ) , dtype=tf.float32 )
def correlationMetric_01mse(x, y, axis=-2):
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
    return tf.constant(1.0, dtype=x.dtype) - corr
def get_model4():
    features_inputs = tf.keras.Input((300, ), dtype=tf.float32)
    
    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dropout(0.4)(feature_x)
    feature_x = layers.Dense(128, activation='swish')(feature_x)
    feature_x = layers.Dropout(0.4)(feature_x)
    feature_x = layers.Dense(64, activation='swish')(feature_x)
    
    x = layers.Concatenate(axis=1)([feature_x])
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(16, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1)(x)
    output = tf.keras.layers.BatchNormalization(axis=1)(output)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001),  loss = correlationLoss, metrics=[correlationMetric])
    return model

def get_model5():
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)
    
    ## feature ##
    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dropout(0.1)(feature_x)
    ## convolution 1 ##
    feature_x = layers.Reshape((-1,1))(feature_x)
    feature_x = layers.Conv1D(filters=16, kernel_size=4, strides=1, padding='same')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU()(feature_x)
    ## convolution 2 ##
    feature_x = layers.Conv1D(filters=16, kernel_size=4, strides=4, padding='same')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU()(feature_x)
    ## convolution 3 ##
    feature_x = layers.Conv1D(filters=64, kernel_size=4, strides=1, padding='same')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU()(feature_x)
    ## convolution 4 ##
    feature_x = layers.Conv1D(filters=64, kernel_size=4, strides=4, padding='same')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU()(feature_x)
    ## convolution 5 ##
    feature_x = layers.Conv1D(filters=64, kernel_size=4, strides=2, padding='same')(feature_x)
    feature_x = layers.BatchNormalization()(feature_x)
    feature_x = layers.LeakyReLU()(feature_x)
    ## flatten ##
    feature_x = layers.Flatten()(feature_x)
    
    x = layers.Dense(512, activation='swish', kernel_regularizer="l2")(feature_x)
    
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse', metrics=['mse', "mae", "mape", rmse])
    return model

kfold = StratifiedKFold(5, shuffle=True, random_state=42)
models = []
best_mse = []
for index, (train_indices, valid_indices) in enumerate(kfold.split(train,investment_id)):
    X_train, X_val = train.iloc[train_indices], train.iloc[valid_indices]
    investment_id_train = investment_id[train_indices]
    y_train, y_val = y.iloc[train_indices], y.iloc[valid_indices]
    investment_id_val = investment_id[valid_indices]
    train_ds = make_dataset_2(X_train, investment_id_train, y_train)
    valid_ds = make_dataset_2(X_val, investment_id_val, y_val, mode="valid")
    model = get_model5()
    checkpoint = keras.callbacks.ModelCheckpoint(f"./ckpt/dnn/dnn_model_5_0330_{index}.tf",monitor="val_loss",save_best_only=True, save_weights_only=True,verbose=1)
    early_stop = keras.callbacks.EarlyStopping(patience=5)
    history = model.fit(train_ds, epochs=30, validation_data=valid_ds, callbacks=[checkpoint, early_stop])
    # models.append(keras.models.load_model(f"./ckpt/dnn_model_0330_{index}.tf"))
    pearson_score = stats.pearsonr(model.predict(valid_ds).ravel(), y_val.values)[0]
    print('Pearson:', pearson_score)
    # pd.DataFrame(history.history, columns=["mse", "val_mse"]).plot()
    # plt.title("MSE")
    # plt.savefig(os.path.join(save_fig_dir,"MSE.png"))
    # plt.show()
    # pd.DataFrame(history.history, columns=["mae", "val_mae"]).plot()
    # plt.title("MAE")
    # plt.savefig(os.path.join(save_fig_dir,"MAE.png"))
    # plt.show()
    # pd.DataFrame(history.history, columns=["rmse", "val_rmse"]).plot()
    # plt.title("RMSE")
    # plt.savefig(os.path.join(save_fig_dir,"RMSE.png"))
    # plt.show()

    #
    # best_mse.append(max(history.history['val_mse']))

    del investment_id_train
    del investment_id_val
    del X_train
    del X_val
    del y_train
    del y_val
    del train_ds
    del valid_ds
    gc.collect() # clean om
    # break


print("finish training...")
