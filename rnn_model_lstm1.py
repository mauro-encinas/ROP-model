# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:01:14 2020

@author: mauro
"""

from logging import info

import numpy as np
from numpy.random import seed
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dense, Dropout, LSTM,
                                         Flatten, GaussianNoise, concatenate)
    
from tensorflow.keras.models import load_model
    
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from data_preparation import step_preparation


def train_model(percent, data_length, memory, predictions, X_prep, X1_prep):
    
    seed(0)
    tf.random.set_seed(0)
  
    #Hyperparameters
    
    memory = memory
    predictions = predictions
    
    newdim = 6
    inc_layer1 = 281
    inc_layer2 = 7
    #gaussian noise now, divided by 1000.
    data_layer1 = 5
    data_layer2 = 1
    dense_layer = 44
    
    #np.arange(139-step, 139+step+1, step) 
    range_max = 1  #DISABLED

    drop1 = 45
    drop2 = 43
    lr = 12
    bs = 51
    ensemble_count = 10

    drop1 = drop1/100
    drop2 = drop2/100
    inc_layer2 = inc_layer2/1000
    lr= lr/10000

    #Data Preparation
    
    RNN_scaler = MinMaxScaler()

    X = RNN_scaler.fit_transform(X_prep) #Scale parameters for Depth,DWOB,etc.

    X1 = RNN_scaler.fit_transform(X1_prep) # Scale ROP values

    X2 = X1[:-memory] #ROP values minus the last 20 rows
    Y = X1[memory:] #ROP values starting at 20 rows

    X_data_1 = step_preparation(X,memory+predictions-1,1) #shift parameters 40 time steps
    X_data_2 = step_preparation(X2,memory-1,1) #shift ROP_1 values 100 time steps

    y = step_preparation(Y,memory-1,1) #shift ROP_2 values 100 time steps
    
     
    border1 = int((data_length)*(percent*0.8)) #80% drilling data
    border2 = int((data_length)*(percent)) #20% drilling data
    border3 = int((data_length)*(percent+0.2))# 20% unknown data, for first case 35% of data total (15+20)
    
    # split into input and outputs
    
    X1_train =  X_data_2[:border1].to_numpy()
    X1_test = X_data_2[border1:border2].to_numpy()
    X1_test2 = X_data_2[border2:border3].to_numpy()
    
    Xdata_train = X_data_1[:border1].to_numpy()
    Xdata_test = X_data_1[border1:border2].to_numpy()
    Xdata_test2 = X_data_1[border2:border3].to_numpy()
    
    y_train,y_test, y_test2 = y[:border1],y[border1:border2], y[border2:border3]
    
    # reshape input [samples, timesteps, features]
    
    X1_train = X1_train.reshape((X1_train.shape[0],X1_train.shape[1],1)) #shape of ROP_1 #,100,1
    X1_test = X1_test.reshape((X1_test.shape[0],X1_test.shape[1],1))
    X1_test2 = X1_test2.reshape((X1_test2.shape[0],X1_test2.shape[1],1))
    
    #shape of  other parameters #,200,6
    Xdata_train_con = Xdata_train.reshape((Xdata_train.shape[0],memory+predictions,newdim)) 
    Xdata_test_con = Xdata_test.reshape((Xdata_test.shape[0],memory+predictions,newdim))
    Xdata_test2_con = Xdata_test2.reshape((Xdata_test2.shape[0],memory+predictions,newdim))
    
    X_train = [X1_train, Xdata_train_con]
    X_test = [X1_test, Xdata_test_con]
    X_test2 = [X1_test2, Xdata_test2_con]
    
    input1 = Input(shape=(memory,1))
    input2 = Input(shape=(memory + predictions,newdim))
    
    # RNN branch
    
    x1 = GaussianNoise(inc_layer2, input_shape=(memory,1))(input1)
   
    x1 = LSTM(units=inc_layer1, kernel_initializer = 'glorot_uniform', recurrent_initializer='orthogonal',
          bias_initializer='zeros', kernel_regularizer='l2', recurrent_regularizer=None,
          bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
          recurrent_constraint=None, bias_constraint=None, return_sequences=False,
          return_state=False, stateful=False)(x1)
    x1 = Dropout(drop1)(x1)
   
    x1 = Model(inputs=input1, outputs=x1)

    # MLP branch
    
    x2 = Dense(data_layer1, input_shape=(memory+predictions,newdim))(input2)
    x2 = Dropout(drop2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(data_layer2)(x2)
    x2 = Model(inputs=input2, outputs=x2)

    combined = concatenate([x1.output, x2.output])

    z = Dense(dense_layer, activation="relu")(combined)
    z = Dense(predictions, activation="linear")(z)

    model = Model(inputs=[x1.input, x2.input], outputs=z)
    
    myadam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    class PlotResuls(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.i += 1

            if (epoch % 14999 == 0) & (epoch > 0):
                print(epoch)

                plt.plot(self.x, np.log(self.losses), label="loss")
                plt.plot(self.x, np.log(self.val_losses), label="val_loss")
                plt.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
                plt.title("Loss")
                plt.legend()
                plt.show();

    plot_results = PlotResuls()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

    
    ens_val_array = np.zeros(ensemble_count)
    ens_test_array = np.zeros(ensemble_count)
    
    for ens_no in range(ensemble_count):
        tf.keras.backend.clear_session()
        mc = ModelCheckpoint(f'best_model_ens_{ens_no}.h5', monitor='val_loss',
                             mode='min', save_best_only=True, verbose=0)
        model = Model(inputs=[x1.input, x2.input], outputs=z)
        model.compile(optimizer=myadam,loss='mean_squared_error')
        history = model.fit(X_train,y_train,validation_data=(X_test, y_test),
                            epochs=2000, verbose=0, batch_size=bs,
                            callbacks=[plot_results, es, mc])

        model = load_model(f'best_model_ens_{ens_no}.h5')
        valresult = np.log(model.evaluate(x= X_test, y=y_test, verbose=0))
        testresult = np.log(model.evaluate(x= X_test2, y=y_test2, verbose=0)) 
        
        ens_val_array[ens_no] = valresult
        ens_test_array[ens_no] = testresult

        
    winner = ens_val_array.argmin()
    model = load_model(f'best_model_ens_{winner}.h5')
    
    info(ens_val_array)
    info(ens_test_array)
    info(f'Validation winner {winner}')

    sample_count = len(X_test2[0])

    y_pred = model.predict(X_test2)
    
    
    ##### Different ensemble, voting ######
    ypred_array = []
    for i in range(ensemble_count):
        model = load_model(f'best_model_ens_{i}.h5')
        y_pred = model.predict(X_test2)
        ypred_array.append(y_pred)

    y_pred = np.average(ypred_array, axis=0)
    
    
    ######## Different ensemble ends here #
    
    y_test_descaled = RNN_scaler.inverse_transform(y_test2)
    y_pred_descaled = RNN_scaler.inverse_transform(y_pred) 
 
    error_matrix = y_pred_descaled-y_test_descaled    

    def rand_jitter(arr):
        stdev = .004*(max(arr)-min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None,
                vmax=None, alpha=None, linewidths=None, verts=None, **kwargs):
        return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c,
                            marker=marker, cmap=cmap, norm=norm, vmin=vmin,
                            vmax=vmax, alpha=alpha, linewidths=linewidths,
                            verts=verts, **kwargs)

    
    
    plt.figure(figsize=(5,5), dpi=200)
    for i in range(sample_count):
        _ = jitter(np.arange(0,memory,1),error_matrix[i], alpha=1,s=0.5,
                    marker=".", c="black")
    plt.title(f"delta, nominal, {percent}")  
    plt.scatter(np.arange(0,memory,1),np.average(np.abs(error_matrix), axis=0),
                marker="o",s=40, alpha=0.7, c="white", zorder=2)
    
    c_array = np.empty(memory, dtype=object)
    aae = np.average(np.abs(error_matrix), axis=0)
    for i in range(memory):
        if aae[i] <= 2.0:
            c_array[i] = "green"
        elif aae[i] <= 4.0:
            c_array[i] = "orange"
        else:
            c_array[i] = "red"
    
    plt.scatter(np.arange(0,memory,1),aae, marker=".",s=20, alpha=1, c=c_array,
                zorder=3, label="Average Absolute Error")
    plt.ylim((-8,8))
    plt.axhline(y=0, xmin=0, xmax=1, linewidth=2, c="black")
    plt.axhline(y=2.0, xmin=0, xmax=1, linewidth=1, c="black")
    plt.axhline(y=4.0, xmin=0, xmax=1, linewidth=1, c="black")
    plt.legend()
    plt.show()

    valresult = np.log(model.evaluate(x= X_test, y=y_test, verbose=0))
    testresult = np.log(model.evaluate(x= X_test2, y=y_test2, verbose=0))
   
    return valresult, testresult, aae, y_test_descaled, y_pred_descaled