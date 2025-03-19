import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16, ResNet50
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define constants
num_classes = 2
image_resize = 224
batch_size = 100
num_epochs = 2
data_path = 'concrete_data_week4'

# Check if dataset exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset folder '{data_path}' not found.")

# Data generators with correct preprocessing for each model
data_generator = ImageDataGenerator(preprocessing_function=vgg16_preprocess)
train_generator = data_generator.flow_from_directory(
    os.path.join(data_path, 'train'),
    target_size=(image_resize, image_resize),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = data_generator.flow_from_directory(
    os.path.join(data_path, 'valid'),
    target_size=(image_resize, image_resize),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build VGG16 model
vgg16_base = VGG16(include_top=False, pooling='avg', weights='imagenet')
vgg16_base.trainable = False  # Freeze base model
model = Sequential([
    vgg16_base,
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
fit_history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator,
    verbose=1
)

# Save model
model.save('classifier_vgg16_model.keras')

# Load models
vgg16Mdl = load_model("classifier_vgg16_model.keras")
vgg16Mdl.summary()

# Check if ResNet model exists before loading
resnet_model_path = "classifier_resnet_model.keras"
if os.path.exists(resnet_model_path):
    resnetMdl = load_model(resnet_model_path)
else:
    raise FileNotFoundError("ResNet model file not found. Ensure you have trained and saved it.")

# Test data generator with appropriate preprocessing
test_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)
test_generator = test_datagen.flow_from_directory(
    os.path.join(data_path, 'test'),
    target_size=(image_resize, image_resize),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate models
resnet_eval = resnetMdl.evaluate(test_generator)
print("ResNet Model Evaluation:", resnet_eval)
vgg16_eval = vgg16Mdl.evaluate(test_generator)
print("VGG16 Model Evaluation:", vgg16_eval)

# Predict and classify
def classify_predictions(model, test_generator, model_name):
    predictions = model.predict(test_generator)
    labels = np.argmax(predictions, axis=1)
    class_mapping = {0: "Negative", 1: "Positive"}
    classified_labels = [class_mapping[label] for label in labels]
    print(f"{model_name} Model classification (first five images):", classified_labels[:5])
    return classified_labels

resnet_classes = classify_predictions(resnetMdl, test_generator, "ResNet")
vgg16_classes = classify_predictions(vgg16Mdl, test_generator, "VGG16")

# Visualization
test_images, _ = next(test_generator)
test_images = test_images[:5]  # Take only first 5 images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i].astype("uint8"))
    plt.title(f"Pred: {resnet_classes[i]}", fontsize=10)
    plt.axis("off")
plt.show()
