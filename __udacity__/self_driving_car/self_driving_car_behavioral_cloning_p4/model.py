import csv
from PIL import Image 
import cv2
import numpy as np

# Data folder names 
data_folder1 = 'data_x/'  # Original/udacity data
data_folder2 = 'data_errors/' # recovery videos
data_folder3 = 'data_smooth_curves/' # more smooth driving
data_folder4 = 'data_curves/' # more curves
data_folder5 = 'data_extra/' # more data

def TrainingData(foldername):
    lines = []
    with open(foldername + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    images = []
    labels_steering = [] 
    i = 0
    for line in lines[1:]:
        print(i)
        i = i + 1
        filename = line[0].split('/')[-1]
        current_path =  foldername + 'IMG/' + filename
        #current_path = base_dir + udacity_data + line[0]
        image = Image.open(current_path)
        images.append(np.array(image))
        labels_steering.append(float(line[3]))


    aug_images, aug_labels = [], []

    for image, label in zip(images, labels_steering):
      aug_images.append(np.array(image))
      aug_labels.append(np.array(label))
      aug_images.append(cv2.flip(np.array(image), 1))
      aug_labels.append(np.array(label) * -1.0)

    X_train = np.array(aug_images)
    y_train = np.array(aug_labels)
    
    return X_train, y_train
    


X_train_1, y_train_1 = TrainingData(data_folder1)

X_train_2, y_train_2 = TrainingData(data_folder2)

X_train_3, y_train_3 = TrainingData(data_folder3)

X_train_4, y_train_4 = TrainingData(data_folder4)
    
X_train_5, y_train_5 = TrainingData(data_folder5)    

# Concatenating data into the training set
X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=0) 
y_train = np.concatenate((y_train_1, y_train_2, y_train_3, y_train_4, y_train_5), axis=0)   
    
    

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout


model = Sequential()

model.add(Cropping2D(cropping = ((70, 25),(0,0)), input_shape = (160,320,3)))

model.add(Lambda(lambda x: x/255.0 - 0.5))

# NVIDIA's convnet 
model.add(Conv2D(24, 5, strides = (2,2), activation = 'relu'))
model.add(Conv2D(36, 5, strides = (2,2), activation = 'relu'))
model.add(Conv2D(48, 5, strides = (2,2), activation = 'relu'))
model.add(Conv2D(64, 3, activation = 'relu'))
model.add(Conv2D(64, 3, activation = 'relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, batch_size= 128, shuffle = True, epochs = 1, verbose=1)

model.save('model.h5')
