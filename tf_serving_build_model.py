

import os
import tensorflow as tf
import keras
# Import the libraries needed for saving models
# Note that in some other tutorials these are framed as coming from tensorflow_serving_api which is no longer correct
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
import keras.backend as K

#load the model
model =tf.keras.models.load_model('tf_serving_keras_resnetbm.h5')

# export_path is a directory in which the model will be created
export_path = os.path.join(
    tf.compat.as_bytes('models/export/{}'.format('resbm_tfserving')),
    tf.compat.as_bytes('1'))

# SavedModelBuilder will create the directory if it does not exist
builder = saved_model_builder.SavedModelBuilder(export_path)

# images will be the input key name
# scores will be the out key name
prediction_signature = predict_signature_def(inputs={'images': model.input},
                                  outputs={'scores': model.output})

#sess.run(tf.global_variables_initializer())
  	
with tf.Session(graph=tf.Graph()) as sess:
# Add the meta_graph and the variables to the builder
	builder.add_meta_graph_and_variables(
	    sess, [tag_constants.SERVING],
	    signature_def_map={
	        'prediction': prediction_signature
	    })
	# save the graph

builder.save()