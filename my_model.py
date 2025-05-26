import keras
import numpy as np
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout,Bidirectional,BatchNormalization,Conv1D,LeakyReLU,MaxPool1D,InputLayer,Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
from keras.src.layers import Reshape
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from scipy.linalg import block_diag

class LSTMRegression(object):


    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size


    def fit(self,X_train,y_train,x_val,y_val):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer

        model.add(Bidirectional(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=0,kernel_regularizer=keras.regularizers.l2(0.01),return_sequences=False))) #Within recurrent layer, include dropout

        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        # model.add(LSTM(units=self.units, dropout=self.dropout,  return_sequences=False))
        # model.add(BatchNormalization())

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, min_lr=0.0001)

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer=optimizer,metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]) #Set loss function and optimizer

        history=model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose,validation_data=(x_val,y_val),callbacks=[reduce_lr],batch_size=self.batch_size) #Fit the model
        self.model=model
        return history


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted



class CNN_lstm(object):
    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size

    def fit(self,X_train,y_train,x_val,y_val):

        """
        Train CNN Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Conv1D(192,1,strides=1,padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(MaxPool1D(2,strides=1))
        model.add(Bidirectional(LSTM(self.units,  dropout=self.dropout,
                                     recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=False)))
        model.add(Dropout(self.dropout))                            
        # model.add(Conv1D(128, 3, strides=1, padding="same"))
        # model.add(LeakyReLU())
        # model.add(BatchNormalization())
        # model.add(MaxPool1D(2))
        # model.add(Flatten())
        # model.add(Dense(100))
        # model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1]))
        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        optimizer=keras.optimizers.Adam(learning_rate=0.001)

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])  # Set loss function and optimizer

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr], batch_size=self.batch_size)  # Fit the model
        self.model = model
        return history

    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted


class LSTMRegression_v2(object):#lstm选择最好验证集上最好的模型进行测试


    """
    Class for the gated recurrent unit (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size


    def fit(self,X_train,y_train,x_val,y_val):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        #Add recurrent layer

        model.add(Bidirectional(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=0,kernel_regularizer=keras.regularizers.l2(0.01),return_sequences=False))) #Within recurrent layer, include dropout

        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        # model.add(LSTM(units=self.units, dropout=self.dropout,  return_sequences=False))
        model.add(BatchNormalization())

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        # optimizer=keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]) #Set loss function and optimizer

        checkpoint = ModelCheckpoint(filepath="/home/maohaodong/BCI/LFP/result/best_model/best_model.hdf5", monitor='val_loss',verbose=1,save_best_only='True',mode='min')

        history=model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose,validation_data=(x_val,y_val),callbacks=[reduce_lr,checkpoint],batch_size=self.batch_size) #Fit the model
        self.model=load_model("/home/maohaodong/BCI/LFP/result/best_model/best_model.hdf5")
        return history


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted




class CNN_lstm_v2(object):#用验证集上表现最好的模型测试-------Hz
    def __init__(self,units=100,dropout=0,num_epochs=100,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size

    def fit(self,X_train,y_train,x_val,y_val):

        """
        Train CNN Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        model.add(InputLayer(input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(Conv1D(192,1,strides=1,padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(MaxPool1D(2,strides=1))
        model.add(Bidirectional(LSTM(self.units,  dropout=self.dropout,
                                     recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=False)))
        model.add(Dropout(self.dropout))                            
        # model.add(Conv1D(128, 3, strides=1, padding="same"))
        # model.add(LeakyReLU())
        # model.add(BatchNormalization())
        # model.add(MaxPool1D(2))
        # model.add(Flatten())
        # model.add(Dense(100))
        # model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1]))
        # model.add(Dense(1))#只预测x或y，如果同时预测改为上面的代码
        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])  # Set loss function and optimizer

        checkpoint = ModelCheckpoint(filepath="/home/BCI_data/LFP_frequency/best_model/best_model.keras", monitor='val_loss',verbose=1,save_best_only='True',mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr,checkpoint], batch_size=self.batch_size)  # Fit the model
        self.model = load_model("/home/BCI_data/LFP_frequency/best_model/best_model.keras")
        return history

    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted



class CNN_lstm_v2_ff(object):#用验证集上表现最好的模型测试------3f-Hz
    def __init__(self,units=100,dropout=0,num_epochs=100,verbose=1,batch_size=128):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.batch_size=batch_size

    def fit(self,X_train,y_train,x_val,y_val):

        """
        Train CNN Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model=Sequential() #Declare model
        model.add(InputLayer(input_shape=(X_train.shape[1],)))
        model.add(Reshape((-1, X_train.shape[1])))
        model.add(Conv1D(192,1,strides=1,padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(MaxPool1D(1,strides=1))
        model.add(Bidirectional(LSTM(self.units,  dropout=self.dropout,
                                     recurrent_dropout=0, kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=False)))
        model.add(Dropout(self.dropout))
        # model.add(Conv1D(128, 3, strides=1, padding="same"))
        # model.add(LeakyReLU())
        # model.add(BatchNormalization())
        # model.add(MaxPool1D(2))
        # model.add(Flatten())
        # model.add(Dense(100))
        # model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1]))
        # model.add(Dense(1))#只预测x或y，如果同时预测改为上面的代码
        model.summary()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)

        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])  # Set loss function and optimizer

        checkpoint = ModelCheckpoint(filepath="/home/BCI_data/LFP_frequency/best_model/best_model.keras", monitor='val_loss',verbose=1,save_best_only='True',mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr,checkpoint], batch_size=self.batch_size)  # Fit the model
        self.model = load_model("/home/BCI_data/LFP_frequency/best_model/best_model.keras")
        return history

    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        return y_test_predicted




from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.activations import softmax

class CNN_lstm_v2_multi(object):
    def __init__(self, units=100, dropout=0, num_epochs=100, verbose=1, batch_size=128, num_classes=3):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_classes = num_classes  # 新增：分类数

    def fit(self, X_train, y_train, x_val, y_val):
        model = Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Conv1D(192, 1, strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(MaxPool1D(2, strides=1))
        model.add(Bidirectional(LSTM(self.units, dropout=self.dropout, recurrent_dropout=0,
                                     kernel_regularizer=keras.regularizers.l2(0.01),
                                     return_sequences=False)))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='softmax'))  # 修改：多分类问题的输出层

        model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')])  # 修改：损失函数和指标

        checkpoint = ModelCheckpoint(filepath="/home/BCI_data/LFP_frequency/best_model/best_model_multi.keras",
                                     monitor='val_loss', verbose=1, save_best_only='True', mode='min')

        history = model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose,
                            validation_data=(x_val, y_val), callbacks=[reduce_lr, checkpoint],
                            batch_size=self.batch_size)

        self.model = load_model("/home/BCI_data/LFP_frequency/best_model/best_model_multi.keras")

        return history

    def predict(self, X_test):
        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted


