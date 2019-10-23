from keras import backend as K
import tensorflow as tf 
from keras.models import model_from_json
import os

#load the model into keras# load json and create model
json_file = open('ResNetbm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("ResNetbm.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

tf.keras.backend.set_learning_phase(0) # all new operations will be in test mode from now on

export_path = os.path.join(
    tf.compat.as_bytes('models/export/{}'.format('resbm_tfserving')),
    tf.compat.as_bytes('1'))

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': loaded_model.input},
        outputs={'probabilities': loaded_model.output})