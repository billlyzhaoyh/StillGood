from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# num_classes is the number of categories your model chooses between for each prediction
num_classes = 10
resnet_weights_path = '/Users/billyzhaoyh/Desktop/StillGood/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
ResNet_bm = Sequential()
ResNet_bm.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
ResNet_bm.add(Dense(num_classes, activation='softmax'))

# The value below is either True or False.  If you choose the wrong answer, your modeling results
# won't be very good.  Recall whether the first layer should be trained/changed or not.
ResNet_bm.layers[0].trainable = True

ResNet_bm.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224

#data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
#use data augmentation techniques
data_generator= ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)


train_generator = data_generator.flow_from_directory(
        '/Users/billyzhaoyh/Desktop/StillGood/Data/Train',
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/Users/billyzhaoyh/Desktop/StillGood/Data/Val',
        class_mode='categorical')

test_generator = data_generator.flow_from_directory(
        '/Users/billyzhaoyh/Desktop/StillGood/Data/Test',
        class_mode='categorical')

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=1),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

ResNet_bm.fit_generator(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks,
        validation_steps=1)

ResNet_bm.evaluate_generator(generator=test_generator)

#save the model to json and ship it out
# serialize model to JSON
import os
model_json = ResNet_bm.to_json()
with open("ResNetbm_v3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
ResNet_bm.save_weights("ResNetbm_v3.h5")
print("Saved model to disk")

import numpy as np
from keras.preprocessing import image
from matplotlib.pyplot import imshow


img_path = '/Users/billyzhaoyh/Desktop/StillGood/Data/Test/Apple/IMG_20190208_090305.jpg'
img = image.load_img(img_path, target_size=(224, 224))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred=ResNet_bm.predict(x)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(pred)
print(predictions)
print(labels)
