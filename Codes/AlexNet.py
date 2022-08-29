import keras
import matplotlib.pyplot as plt 
import numpy as np
import energyusage
import time
from sklearn.utils.multiclass import unique_labels
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
np.random.seed(1000)

plt.style.use('fivethirtyeight')
start_time = time.time()
def AlexNet():
	#Instantiation
	AlexNet = Sequential()

	#1st Convolutional Layer
	AlexNet.add(Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#2nd Convolutional Layer
	AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#3rd Convolutional Layer
	AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))

	#4th Convolutional Layer
	AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))

	#5th Convolutional Layer
	AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

	#Passing it to a Fully Connected layer
	AlexNet.add(Flatten())
	# 1st Fully Connected Layer
	AlexNet.add(Dense(4096, input_shape=(32,32,3,)))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	# Add Dropout to prevent overfitting
	AlexNet.add(Dropout(0.4))

	#2nd Fully Connected Layer
	AlexNet.add(Dense(4096))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	#Add Dropout
	AlexNet.add(Dropout(0.4))

	#3rd Fully Connected Layer
	AlexNet.add(Dense(1000))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('relu'))
	#Add Dropout
	AlexNet.add(Dropout(0.4))

	#Output Layer
	AlexNet.add(Dense(10))
	AlexNet.add(BatchNormalization())
	AlexNet.add(Activation('softmax'))

	#Model Summary
	AlexNet.summary()

	AlexNet.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])

	#Keras library for CIFAR dataset
	from keras.datasets import cifar10
	(x_train, y_train),(x_test, y_test)=cifar10.load_data()

	#Train-validation-test split
	from sklearn.model_selection import train_test_split
	x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=.3)

	#Dimension of the CIFAR10 dataset
	print((x_train.shape,y_train.shape))
	print((x_val.shape,y_val.shape))
	print((x_test.shape,y_test.shape))

	#Since we have 10 classes we should expect the shape[1] of y_train,y_val and y_test to change from 1 to 10
	y_train=to_categorical(y_train)
	y_val=to_categorical(y_val)
	y_test=to_categorical(y_test)

	#Verifying the dimension after one hot encoding
	print((x_train.shape,y_train.shape))
	print((x_val.shape,y_val.shape))
	print((x_test.shape,y_test.shape))

	#Image Data Augmentation
	from keras.preprocessing.image import ImageDataGenerator

	train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1 )

	val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)

	test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True,zoom_range=.1)

	#Fitting the augmentation defined above to the data
	train_generator.fit(x_train)
	val_generator.fit(x_val)
	test_generator.fit(x_test)

	#Learning Rate Annealer
	from keras.callbacks import ReduceLROnPlateau
	lrr= ReduceLROnPlateau(   monitor='val_accuracy',   factor=.01,   patience=3,  min_lr=1e-5) 

	#Defining the parameters
	batch_size= 32
	epochs= 250
	train_size = 1  # The percentage of the training set to be used during training (0.0 - 1.0) 
    
	 # apply the train_size of the trainset

	n_elements = x_train.shape[0]
	end_index = int(np.floor(n_elements * train_size))
	x_train = x_train[:end_index]
	y_train = y_train[:end_index]

	#Training the model
	AlexNet.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size), epochs = epochs, steps_per_epoch = x_train.shape[0]//batch_size, validation_data = val_generator.flow(x_val, y_val, batch_size=batch_size), validation_steps = 250, callbacks = [lrr], verbose=1)

	#Plotting the training and validation loss

	f,ax=plt.subplots(2,1) #Creates 2 subplots under 1 column

	#Assigning the first subplot to graph training loss and validation loss
	ax[0].plot(AlexNet.history.history['loss'],color='b',label='Training Loss')
	ax[0].plot(AlexNet.history.history['val_loss'],color='r',label='Validation Loss')

	#Plotting the training accuracy and validation accuracy
	ax[1].plot(AlexNet.history.history['accuracy'],color='b',label='Training  Accuracy')
	ax[1].plot(AlexNet.history.history['val_accuracy'],color='r',label='Validation Accuracy')

	plt.legend()

	# Calculate the accuracy

	print("\nCalculating the accuracy over the testset:")
	accuracy = AlexNet.evaluate(x_test, y_test, verbose=1)
	print(f"accuracy: {accuracy[1]}\n")


	#Prediction for testset

	#y_pred=AlexNet.predict(x_test)
	#y_true=np.argmax(y_test,axis=1)


	class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


	# make predictions over the testset
	plt.figure()
	plt.imshow(x_test[0])
	plt.colorbar()
	plt.grid(False)
	plt.show()
	predictions = AlexNet.predict(x_test)
	predictions
	list_index = [0,1,2,3,4,5,6,7,8,9]
	x = predictions
	for i in range(10):
	  for j in range(10):
	    if x[0][list_index[i]] > x[0][list_index[j]]:
             temp = list_index[i]
             list_index[i] = list_index[j]
             list_index[j] = temp
	#Show the sorted labels in order from highest probability to lowest
	print("The sorted labels in order from highest probability to lowest")
	for i in range(10):
	  print(i,":",class_names[list_index[i]])
	  
	print("Prediction for the first image(which is a cat):")
	predictions[0]
	print(class_names[np.argmax(predictions[0])])

	#Test with personal image
	new_image = plt.imread("horse-3.jpg") #Read in the image (3, 14, 20)
	img = plt.imshow(new_image)
	from skimage.transform import resize
	resized_image = resize(new_image, (32,32,3))
	plt.imshow(resized_image)
	predictions = AlexNet.predict(np.array( [resized_image] ))
	predictions
	list_index = [0,1,2,3,4,5,6,7,8,9]
	x = predictions
	for i in range(10):
	  for j in range(10):
	    if x[0][list_index[i]] > x[0][list_index[j]]:
	      temp = list_index[i]
	      list_index[i] = list_index[j]
	      list_index[j] = temp
	#Show the sorted labels in order from highest probability to lowest
	print("The sorted labels in order from highest probability to lowest")
	for i in range(10):
	  print(i,":",class_names[list_index[i]])
	  
	print("Prediction for the imported image(which is a Horse):")
	predictions[0]
	print(class_names[(list_index[0])])

#energyusage.evaluate(AlexNet1)
AlexNet()
print("--- %s seconds ---" % (time.time() - start_time))
