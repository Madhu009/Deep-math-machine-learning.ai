
# coding: utf-8

# In[1]:

from keras.datasets import cifar10
(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()


# # Keras

# In[2]:

print(X_train.shape)
print(X_test.shape)


# In[3]:

#print(X_train[0])


# In[4]:

#we have pixel values here let's convert into between 0 and 1
X_train=X_train/255.0
X_test=X_test/255.0


# In[5]:

#X_train[0]


# In[6]:

#Y_train[0]


# In[7]:

# create a one hot vector for label
from keras.utils import np_utils

Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)


# In[8]:

#Y_train[0]


# In[9]:

#build the cnn model 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.constraints import maxnorm
from keras.layers import Dropout
from keras.layers import Flatten


import numpy as np
def model():
    model=Sequential()
    model.add(Convolution2D(32,3,3,activation='relu',input_shape=(3,32,32),border_mode='same',W_constraint=maxnorm(3)))
    
    #model.add(Dropout(0.2))
    model.add(Convolution2D(32,3,3,activation='relu',input_shape=(3,32,32),border_mode='same',W_constraint=maxnorm(3)))
    
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(512,activation='relu',W_constraint=maxnorm(3)))
    #model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    
    return model
    


# In[10]:

epochs = 10
lrate = 0.01
decay = lrate/epochs
from keras.optimizers import SGD
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model=model()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# In[ ]:

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:

#save model
jsonFile=model.to_json()
with open('cifar10.json','w') as file:
    file.write(jsonFile)
model.save_weights('cifar10.h5')

