import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import backend as K
from keras import regularizers 
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from sklearn.base import clone
from scipy import stats
from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold, GroupKFold
from tqdm import tqdm
from tensorflow.python.ops import math_ops
import ubiquant
env = ubiquant.make_env()
iter_test = env.iter_test()
import pickle
investment_ids = pickle.load(open("../input/ubi-dnn-ensemble-features/investment_id_dnn_0329.pkl","rb"))

# investment_ids = list(investment_id.unique())
investment_id_size = len(investment_ids) + 1
investment_id_lookup_layer = layers.IntegerLookup(max_tokens=investment_id_size)
investment_id_lookup_layer.adapt(pd.DataFrame({"investment_ids":investment_ids}))
n_features = 300
features = [f'f_{i}' for i in range(n_features)]

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


def preprocess_test(investment_id, feature):
    return (investment_id, feature), 0

def preprocess_test_s(feature):
    return (feature), 0

def make_test_dataset(feature, investment_id, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(((investment_id, feature)))
    ds = ds.map(preprocess_test)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def make_test_dataset2(feature, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices(((feature)))
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds

def inference(models, ds):
    y_preds = []
    for model in models:
        y_pred = model.predict(ds)
        y_preds.append(y_pred)
    return np.mean(y_preds, axis=0)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)


def make_test_dataset3(feature, batch_size=1024):
    ds = tf.data.Dataset.from_tensor_slices((feature))
    ds = ds.map(preprocess_test_s)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def infer(models, ds):
    y_preds = []
    for model in models:
        y_pred = model.predict(ds)
        y_preds.append((y_pred-y_pred.mean())/y_pred.std())
    
    return np.mean(y_preds, axis=0)

# load ckpt
models3 = []

for index in range(10):
    model = get_model4()
    model.load_weights(f"../input/model10mse/model_{index}")
    models3.append(model)


for (test_df, sample_prediction_df) in iter_test:
    ds = make_test_dataset(test_df[features], test_df["investment_id"])
    p1 = inference(models, ds)
    ds2 = make_test_dataset2(test_df[features])
    p2 = inference(models2, ds2)
    ds3 = make_test_dataset3(test_df[features])
    p3 = infer(models3, ds3)
    sample_prediction_df['target'] = p1 * 0.29 + p2 * 0.64 + p3 * 0.07
    env.predict(sample_prediction_df) 
    display(sample_prediction_df)