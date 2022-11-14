import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import train_test_split
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
import pandas as pd

dataset = pd.read_csv("refrigerator_10_lac.csv")
print(dataset.head())

x = dataset["main"].values
y = dataset["xx"].values

xx = np.array(x)
yy = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(xx,yy,test_size = 0.15,random_state = 1)

def sequesnceGenerator(arr,n):
    i=0
    arr1 = []
    temp = []
    while(i < len(arr)):
        if(i%n == 0 and i != 0):
            arr1.append(temp)
            temp = [arr[i]]
        else:
            temp.append(arr[i])
        i+=1
    #arr1.append(temp)
    return arr1

X_TRAIN=np.array(sequesnceGenerator(x_train,8))
Y_TRAIN=np.array(sequesnceGenerator(y_train,8))
X_TEST=np.array(sequesnceGenerator(x_test,8))
Y_TEST=np.array(sequesnceGenerator(y_test,8))

print(X_TRAIN.shape)

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(8))),
# model.add(Dense(200, input_dim=200, activation='relu'))
# model.add(Dense(200, input_dim=200, activation='relu'))
# model.add(Dense(200, input_dim=200, activation='relu'))
# model.add(Dense(8, activation='linear'))


cnn = models.Sequential([
    layers.Conv1D(filters=64, kernel_size=1, activation='relu',input_shape=(8,1)),
    layers.Dropout(0.5),

    layers.Conv1D(filters=32, kernel_size=1, activation='relu'),
    layers.Conv1D(filters=16, kernel_size=1, activation='relu'),
    layers.MaxPooling1D(pool_size=1, name="MaxPooling1D"),

    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(8, activation='linear')
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

cnn.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

#model.fit_generator(dataLoader(xx,yy,5), epochs = 10)
history = cnn.fit(X_TRAIN, Y_TRAIN, epochs=20, batch_size=8,validation_split=0.15,validation_data=None,verbose=1)

file_name = "testModelRefrigerator.h5"
cnn.save(file_name)

print(cnn.predict(X_TEST,batch_size=8))
print("---------------------------------------")
print(Y_TEST)


# plot metrics
pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['accuracy'])
# pyplot.plot(history.history['mean_absolute_error'])
#pyplot.plot(history.history['mean_absolute_percentage_error'])
# pyplot.plot(history.history['cosine_proximity'])
pyplot.show()

#score = model.evaluate(X_test, y_test, verbose = 0) 

# print('Test loss:', score[0]) 
# print('Test accuracy:', score[1])