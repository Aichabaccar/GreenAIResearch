import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import energyusage
import time
import keras
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import visualkeras
start_time = time.time() 
plt.style.use('fivethirtyeight')


def CNN():
    from keras.datasets import cifar10
    print("\n -----test3-----\n")
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_images.shape
    
    #Get the shape of x_train
    print("Training images shape", train_images.shape)
    #Get the shape of y_train
    print("Training labels shape", train_labels.shape)
    #Get the shape of x_train
    print("Test images shape", test_images.shape)
    #Get the shape of y_train
    print("Test labels shape", test_labels.shape)
    
   
    train_images = train_images / 255.0
    test_images = test_images / 255.
     
     
    # setting the parameters
    epochs = 1 # How many passes over the data should be made during training
    batch_size = 1  # The number of training examples utilized in one iteration
    verbosity = 1     # Whether to print each epoch step on the screen
    train_size = 1 # The percentage of the training set to be used during training (0.0 - 1.0) 
    
    
    #Learning Rate Annealer
    lrr= ReduceLROnPlateau(   monitor='val_accuracy',   factor=.01,   patience=3,  min_lr=1e-5) 
     
    # apply the train_size of the trainset
    n_elements = train_images.shape[0]
    end_index = int(np.floor(n_elements * train_size))
    X_train = train_images[:end_index]
    y_train = train_labels[:end_index]
    
     
     # CNN model
     # building the model

    model = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
    ])
    
    model.summary()
    

    visualkeras.layered_view(model).show() # display using your system viewer
    visualkeras.layered_view(model, to_file='output.png') # write to disk
    visualkeras.layered_view(model, to_file='output.png').show() # write and show
    visualkeras.layered_view(model)
     
    #Compiling the model
    model.compile(optimizer='adam', 
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
      metrics=['accuracy'])
    #Add validation set
    
    
    #Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, steps_per_epoch = X_train.shape[0]//batch_size, validation_split= 0.1, callbacks = [lrr], verbose=verbosity)
    
     
    
    # Calculate the accuracy
    
    print("\nCalculating the accuracy over the testset:")
    accuracy = model.evaluate(test_images, test_labels, verbose=1)
    print(f"accuracy: {accuracy[1]}\n")
    
    # make predictions over the testset
    print("Prediction with the first image of the testset:")
    predictions = model.predict(test_images)
    y_pred_labels = [np.argmax(label) for label in predictions]
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
    print("The sorted labels in order from highest probability to lowest for the personal image")
    for i in range(10):
      print(i,":",class_names[list_index[i]])
      
    print("Prediction for the first image(which is a cat):")
    predictions[0]
    print(class_names[np.argmax(predictions[0])])


    # build the classification report
    print("Classification report:")
    print(classification_report(test_labels, y_pred_labels, target_names=class_names))

    # build confusion matrix
    print("Confusion matrix:")
    cm = confusion_matrix(test_labels, y_pred_labels)
    print(cm)    
   
  
    #To save this model 
    model.save('my_model_CNN_BIG_TEST.h5')

#energyusage.evaluate(CNN)
CNN()
print("--- %s seconds ---" % (time.time() - start_time))
