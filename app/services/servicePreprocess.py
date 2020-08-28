from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

def invert(image):
	image = ImageOps.invert(image)
	return image

def showImg(x):
	print(x.shape)
	plt.imshow(x[0].numpy().squeeze(), cmap='gray_r')
	plt.show()
	
def preprocess(image):
	data = invert(image)
	x = TF.resize(data, (28, 28))
	x = TF.to_grayscale(x)
	x = TF.to_tensor(x)
	x = TF.normalize(x, (0.5,), (0.5,))
	x.unsqueeze_(0)
	# showImg(x)
	return x