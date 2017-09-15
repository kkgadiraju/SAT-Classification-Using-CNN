import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.metrics import Top_k 
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf

tf.reset_default_graph()
tflearn.init_graph(seed=100)
tf.set_random_seed(100)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=90.)

acc = Accuracy()

network = input_data(shape=[None, 28, 28, 4],data_augmentation=img_aug)
# Conv layers
network = conv_2d(network, 64, 3, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_1', weights_init = 'Xavier')
#network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 128, 3, strides=1, activation='relu', regularizer='L2',name = 'conv1_3_3_2', weights_init = 'Xavier')
#network = max_pool_2d(network, 2, strides=2)
#network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_3')
#network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 256, 3, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_4', weights_init = 'Xavier')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 512, 3, strides=1, activation='relu', regularizer='L2', name = 'conv1_3_3_5', weights_init = 'Xavier')
network = max_pool_2d(network, 2, strides=2)
# Fully Connected Layer 
network = fully_connected(network, 512, activation='relu')
# Dropout layer
network = dropout(network, 1)
# Fully Connected Layer 
#network = fully_connected(network, 512, activation='relu')
# Dropout layer
#network = dropout(network, 1)
# Fully Connected Layer
network = fully_connected(network, 6, activation='softmax')
# Final network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0007, metric=acc)

# The model with details on where to save
# Will save in current directory
model = tflearn.DNN(network, checkpoint_path='/scratch/slums/bl-slums/model/model-', best_checkpoint_path='/scratch/slums/bl-slums/model/best-model-', tensorboard_verbose=1)



