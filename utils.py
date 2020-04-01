import warnings
warnings.filterwarnings('ignore')

# data processing
import pandas as pd
import numpy as np

# image processing
from PIL import Image


# tf and keras
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K


# dataset processing, ml models and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import scipy
import glob


__all__ = [
'read_csv',
'remove_transparency',
'preprocess_img',
'populate_images',
'convert_to_one_hot_encode',
'one_hot_encode_to_char',
'convert_pred_list_ohe_to_labels',
'get_df_split',
'gen_x_y_train_test_stratified_1df',
'process_x_y_train_test_stratified_2df',
'get_label_count_df',
'get_label_count_train_test_dfs',
'dir_',
'model_dir',
'data_dir',
'processed_data_dir',
]

dir_ = 'HASYv2/'
model_dir = 'trained_models/'
data_dir = 'data/'
processed_data_dir ='processed_data/'

def read_csv(path):
    return pd.read_csv(path)


# Image Preprocessing
def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency 
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL 
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

def preprocess_img(path):
    # Open Image
    im = Image.open(dir_ + path)
    
    # Resize image to 32 by 32
    if im.size != (32,32):
        im = im.resize((32,32))
        
    # Convert image to a single greyscale channel
    im = remove_transparency(im).convert('L')
    
    # Convert image to numpy array
    I = np.asarray(im)
    
    #Close image
    im.close()
    
    return I


def populate_images(dataset):
    temp = []
    for i in range(len(dataset)):
        path = dataset.iloc[i]['path']
        pathsplit = path.split('/')
        if len(pathsplit) > 2:
            path = '/'.join([pathsplit[-2],pathsplit[-1]])
        img = preprocess_img(path)
        temp.append(img)
    dataset['img'] = [i for i in temp]
    return dataset

def convert_to_one_hot_encode(data, no_categories):
    data = np.array(data).reshape(-1)
    print('len of dataset', len(data))
    return np.eye(no_categories)[data]

# to process output to the value
# returns a list with all the categories with more than 50% accuracy
def one_hot_encode_to_char(arr, threshold=0.5, get_max=True):
    result = []
    val = 0
    for i in range(len(arr)):
        if arr[i] >= threshold:
            result.append((val, arr[i]))
        val +=1
    _max = []
    high = 0
    if get_max:
        for i in result:
            if i[1] > high:
                _max = [i[0]]
                high = i[1]
        return _max
    else:
        return [i[0] for i in result]

def convert_pred_list_ohe_to_labels(pred_data, threshold=0.5, get_max=True):
    result = []
    for i in range(len(pred_data)):
        val = one_hot_encode_to_char(pred_data[i], threshold=threshold, get_max=get_max)
        if len(val) > 0:
            if get_max == True:
                result.append(val[0])
            else:
                result.append(val)
        else:
            result.append(None)
            print(":( :( :(")
    return result


# Dataset Splitting
# Stratified Train Test Split (new function)
def get_df_split(ds, stratify_col, test_size=0.2):
    _train, _test = train_test_split(ds, test_size=test_size, stratify=ds[stratify_col])
    return _train, _test

# function to split whole dataset at once (old function)
def gen_x_y_train_test_stratified_1df(dataset, input_shape, test_size=0.2):
    x = np.array(list(dataset['img']))
    y = np.array(list(dataset['symbol_id_ohe']))
    x = x.reshape((x.shape[0],1,input_shape[1],input_shape[2]))
    # Normalize data to 0-1
    x = x.astype("float32") / 255.0
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size,  stratify=y)
    return X_train, X_test, y_train, y_test

# function to process already split data
def process_x_y_train_test_stratified_2df(_tr, _ts, input_shape):
    # train df
    X_train = np.array(list(_tr['img']))
    y_train = np.array(list(_tr['symbol_id_ohe']))
    X_train = X_train.reshape((X_train.shape[0],1,input_shape[1],input_shape[2]))
    # Normalize data to 0-1
    X_train = X_train.astype("float32") / 255.0
    # test df
    X_test = np.array(list(_ts['img']))
    y_test = np.array(list(_ts['symbol_id_ohe']))
    X_test = X_test.reshape((X_test.shape[0],1,input_shape[1],input_shape[2]))
    # Normalize data to 0-1
    X_test = X_test.astype("float32") / 255.0
    
    return X_train, X_test, y_train, y_test

# Dataset metrics
# generate label counts for dataframe and list
def get_label_count_df(df_train, df_test, sym_list):
    train_labels_count = {}
    test_labels_count = {}
    perc_labels_count = {}
    for i in sym_list:
        train_labels_count[i] = 0
        test_labels_count[i] = 0
    for i in range(len(df_train)):
        train_labels_count[df_train.loc[i,'symbol_id']] += 1
    for i in range(len(df_test)):
        test_labels_count[df_test.loc[i,'symbol_id']] += 1
    for i in sym_list:
        perc = (train_labels_count[i] / (train_labels_count[i] + test_labels_count[i])) * 100
        perc_labels_count[i] = (train_labels_count[i], test_labels_count[i], round(perc,2))
    return perc_labels_count

def get_label_count_train_test_dfs(df_train, df_test):
    train_labels_count = {}
    test_labels_count = {}
    perc_labels_count = {}
    train_syms = df_train['symbol_id'].unique()
    test_syms = df_test['symbol_id'].unique()
    sym_list = np.unique(np.concatenate([train_syms, test_syms], axis=0))
    for i in sym_list:
        train_labels_count[i] = 0
        test_labels_count[i] = 0
    for i in range(len(df_train)):
        train_labels_count[df_train.loc[i,'symbol_id']] += 1
    for i in range(len(df_test)):
        test_labels_count[df_test.loc[i,'symbol_id']] += 1
    for i in sym_list:
        perc = (train_labels_count[i] / (train_labels_count[i] + test_labels_count[i])) * 100
        perc_labels_count[i] = (train_labels_count[i], test_labels_count[i], round(perc,2))
    return perc_labels_count

def get_label_count_list(lst_data, sym_list):
    labels_count = {}
    for i in sym_list:
        labels_count[i] = 0
    for i in range(len(lst_data)):
        j = one_hot_encode_to_char(lst_data[i])[0]
        labels_count[j] += 1
    return labels_count




