import requests
import json
import sys
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np

def prepare_image(image, target=(224, 224)):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

image_path = './Data/Test/Lime/IMG_20190121_100153.jpg'
image = Image.open(image_path)
image=np.array(image)
#image = prepare_image(image, target=(224, 224))

labels={0: 'Carrot', 1: 'Garlic', 2: 'Lemon', 3: 'Lime', 4: 'Onion', 5: 'Potato'}
ingredients={'Carrot':336,'Garlic':356,'Lemon':266,'Lime':266,'Onion':329,'Potatoes':326}

# setup the request
url = "http://localhost:8501"
full_url = f"{url}/v1/models/resbm_tfserving:predict"

data = {"instances":[{"input_image":image.tolist()}]}
data = json.dumps(data)
try:
    response = requests.post(full_url,data=data)
    response = response.json()
    highest_index = np.argmax(response['predictions'])
    item=labels[int(highest_index)]
    
    print(ingredients[item])
except:
    print(sys.exc_info()[0])