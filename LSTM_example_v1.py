
# give credit for this link: # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
from pandas import read_csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
dataset = read_csv('pollution_s.csv')

## fix missing data:
def fill_missing_values(df):
    ''' This function imputes missing values with median for numeric columns 
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    # select missing data
    missing = missing[missing > 0]
    for column in list(missing.index):
        if df[column].dtype == 'object':
            # if it's an object, fill that with the *most common* object in that column 
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True)
#%%
fill_missing_values(dataset)



## encoder:
from sklearn.preprocessing import LabelEncoder
def impute_cats(df):
    '''This function converts categorical and non-numeric 
       columns into numeric columns to feed into a ML algorithm'''
    # Find the columns of object type along with their column index
    # only select columns with obejcts
    object_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    # return the index for columns with object
    object_cols_ind = []
    for col in object_cols:
        object_cols_ind.append(df.columns.get_loc(col))


    
    # Encode the categorical columns with numbers    
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    # It's still an object but this time with index from 0 to num_features-1
    label_enc = LabelEncoder()
    for i in object_cols_ind:
        df.iloc[:,i] = label_enc.fit_transform(df.iloc[:,i])
    
impute_cats(dataset)



from matplotlib import pyplot
import matplotlib.pyplot as plt
# load dataset
values = dataset.values
label_names = list(dataset.columns)
index_names = label_names[1:]
index_names

# index them
import numpy as np
time_array = list(range(0,len(dataset)))
print(len(time_array))
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
for i in range(len(index_names)):
    plt.subplot(len(index_names),1,i+1)
    plt.plot(time_array,dataset[index_names[i]],"k")
    pyplot.title(index_names[i], y=0.5, loc='right')


print("There are %d hours inside"%(len(time_array)))

##!!!!!!!!! normalized to 0 to 1 for both train_x and train_y

X = dataset[index_names].values[:-1,:]
y = dataset[index_names[0]][1:].values

# re-scale to 0-1:
# FOR BOTH X AND Y

X = (X-np.nanmin(X,axis=0))/(np.nanmax(X,axis=0)-np.nanmin(X,axis=0))


X = X.reshape([-1,1,8])

y = (y-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))

# split into trian test set with 8:2
n_train_hours = int(0.8*len(time_array))

X_train = X[:n_train_hours,:]
X_test = X[n_train_hours:,:]

y_train = y[:n_train_hours]
y_test = y[n_train_hours:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

## define a network:
# We will define the LSTM with 50 neurons in the first hidden layer and 1 neuron in the output layer for predicting pollution. The input shape will be 1 time step with 8 features.
import tensorflow as tf
from tensorflow.keras import Sequential
import keras

# design network
model = tf.keras.Sequential([
  tf.keras.layers.LSTM(50,input_shape=(X_train.shape[1],X_train.shape[2])),  # must declare input shape
  tf.keras.layers.Dense(1)
])

model.compile(loss='mae', optimizer='adam')
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

y_pre = model.predict(X_test)


plt.plot(y_pre,y_test,"k.")

mse = ((y_pre.ravel()-y_test)**2).mean(axis=0)/np.nanmedian(y_test)
print("Mean square error =%.6f"%mse)
