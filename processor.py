import numpy as np
from keras.preprocessing.image import img_to_array, load_img


# stack a set of images together as different channels
def stack_image(frames, target_shape):

	cnt = 0
	for image in frames:

		# load image
		h, w, _ = target_shape
		
		try:
			image = load_img(image, target_size=(h, w), grayscale=True)
		except IOError:
			print("\n")
			print("\n")
			print(image)
			print("\n")
			print("\n")


		# Turn it into numpy, normalize and return.
		img_arr = img_to_array(image)

		# stack images together
		if cnt == 0:
			x = (img_arr / 255.).astype(np.float32)
		else:
			x = np.dstack((x, (img_arr / 255.).astype(np.float32)))

		# update the counter
		cnt += 1

	return x


# each image is defined temporally separated
def temporal_image(image, target_shape):

	h, w, _ = target_shape
	try:
		image = load_img(image, target_size=(h, w), grayscale=True)
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