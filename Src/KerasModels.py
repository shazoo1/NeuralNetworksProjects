from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.applications.densenet import DenseNet121
#from keras.applications.densenet import DenseNet169
#from keras.applications.densenet import DenseNet201
from keras.applications.densenet import preprocess_input, decode_predictions
import PIL.Image

import numpy as np
from keras_preprocessing import image

#Load pre-trained model from Internetto
model121 = DenseNet121(weights='imagenet')

#Load an image for recognition
img_path = '../Data/PretrainedModel/1.png'
#Should be of the size described in a documentation of the model, i.e. 224x224 for DenseNet
img = image.load_img(img_path, target_size=(224, 224))
#Convert image into numpy array of pixels
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#Prepare input for recognition
x = preprocess_input(x)

preds121 = model121.predict(x)
# decode the results into a list of tuples (class, description, probability)
#  (one such list for each sample in the batch)
decoded121 = decode_predictions(preds121, top=5)
for i in range(0, len(decoded121[0])):
    # retrieve the most likely result, e.g. highest probability
    #Change variable in order to print other model's result
    labelTemp = decoded121[0][i]
    # print the classification
print('%s (%.2f%%)\n' % (labelTemp[1], labelTemp[2]*100))
