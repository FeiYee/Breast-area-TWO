import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter, maximum_filter, minimum_filter
from skimage import img_as_float


filter_mapping = {
	'Gaussian' : gaussian_filter,
	'Median' : median_filter,
	'Max' : maximum_filter,
	'Min' : minimum_filter,

}

def Gaussian_UM(filename, filter, radius, amount):
	image = imageio.imread(filename)
	image = img_as_float(image) 
	blurred_image = filter_mapping[filter](image, sigma=radius)
  
    ## print()

	mask = image - blurred_image 
	sharpened_image = image + mask * amount

	# print(mask)
	# print(sharpened_image)
	# print(mask)

	sharpened_image = np.clip(sharpened_image, float(0), float(1))
	sharpened_image = (sharpened_image*255).astype(np.uint8)

	return sharpened_image

