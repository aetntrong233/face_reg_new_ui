from keras.models import load_model
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.models import Model


# with tf.device('/device:CPU:0'):
#     base_model = load_model(r'train\200_09602_32119\check_point.h5')
#     base_model.save_weights(r'storage\model\feature_extraction_model\base.h5')
#     emb_layer = base_model.layers[-2]
#     output = emb_layer.output
#     # output = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False)(emb_layer.output)
#     model = Model(base_model.inputs, output)
#     model.save_weights(r'storage\model\feature_extraction_model\200_09602_32119.h5')
#     model.summary()


base_model = load_model(r'train\200_0.95612_0.1466\check_point.h5')
model = Model(base_model.inputs, base_model.layers[-2].output)
model.save_weights(r'storage\model\feature_extraction_model\weights.h5')
