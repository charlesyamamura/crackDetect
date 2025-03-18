
'wget  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip'

'unzip concrete_data_week4.zip'

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

num_classes = 2
image_resize = 224 #image size adjusted to 224
batch_size_training = 100
batch_size_validation = 100

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator.flow_from_directory(
                'concrete_data_week4/train',
                target_size=(image_resize, image_resize), 
                batch_size=batch_size_training, 
                class_mode='categorical')
validation_generator = data_generator.flow_from_directory(
                'concrete_data_week4/valid',
                target_size=(image_resize, image_resize), 
                batch_size=batch_size_validation, 
                class_mode='categorical')

model = Sequential()
model.add(VGG16(include_top=False, pooling='avg', weights='imagenet',))
model.layers[0].trainable = False
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 2
fit_history = model.fit(
            train_generator, 
            epochs=num_epochs, 
            validation_data=validation_generator, 
            verbose=1)

model.save('classifier_vgg16_model.keras')


import tensorflow as tf
from tensorflow.keras.models import load_model
vgg16Mdl = load_model("classifier_vgg16_model.keras")
vgg16Mdl.summary()

resnetMdl = load_model("classifier_resnet_model.keras")

from tensorflow.keras.applications.resnet50 import preprocess_input

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    'concrete_data_week4/test',
    target_size=(image_resize, image_resize),  
    batch_size=100,  
    class_mode='categorical',  
    shuffle=False  
)

resnet_eval = resnetMdl.evaluate(test_generator)
print("ResNet Model Evaluation:", resnet_eval)

vgg16_eval = vgg16Mdl.evaluate(test_generator)
print("VGG16 Model Evaluation:", vgg16_eval)

import numpy as np
resnet_pred = resnetMdl.predict(test_generator) # predict_generator() is deprecated; used predict() instead
resnet_labels = np.argmax(resnet_pred, axis=1)
class_mapping = {0: "Negative", 1: "Positive"}
resnet_classes = [class_mapping[label] for label in resnet_labels]
print("ResNet Model classification (first five images):", resnet_classes[:5])

vgg16_pred = vgg16Mdl.predict(test_generator)
vgg16_labels = np.argmax(vgg16_pred, axis=1)
class_mapping = {0: "Negative", 1: "Positive"}
vgg16_classes = [class_mapping[label] for label in vgg16_labels]
print("VGG Model classification (first five images):", vgg16_classes[:5])

import matplotlib.pyplot as plt
test_images, _ = next(test_generator)  
plt.figure(figsize=(10, 5))
for i in range(5):  
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i].astype("uint8"))  
    plt.title(f"Pred: {resnet_classes[i]}", fontsize=10)
    plt.axis("off")
plt.show()
