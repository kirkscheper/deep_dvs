import numpy as np
from keras.preprocessing.image import img_to_array, load_img

# stack a set of images together as different channels
def load_image(image_path, target_shape):
	# load image

	h, w, _ = target_shape
	
	try:
		image = load_img(image_path, target_size=(h, w), grayscale=True)
	except IOError:
		print("\n")
		print("\n")
		print(image)
		print("\n")
		print("\n")

	# Turn it into numpy, normalize and return.
	img_arr = img_to_array(image)
	x = (img_arr / 255.).astype(np.float32)

	return x