# FLOCNN
FLOCNN - Flower Object CNN model

Create the convolutional base/layer

The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.

These are filters (stuff that the image is multiplied so features within images would come forward and you can spot said features). The filters are LEARNED like other parameters in the neurons.

As our image is feed into the convolutional layer, a number of randomly initialized filters will pass over the image
Over time the filters that give us the image outputs that give the best matches will be learned. This is called feature extraction.
Pooling: which groups up the pixels in the image and filters them down to a subset. MaxPooling 2x2 will group the image into 2x2 pixel sections, and simply pick the largest pixel value. This will leave an image a quarter of the size and its features can be enhanced.

CNN take inputs of tensors of shape like image_height, image_width, color channel) image_height = 32 pixels image_weight = 32 pixels color_chanel = 3 (this is because there are 3 chanels -> r + g + b)

We add a convolution layer on top of it that takes the input of (32, 32, 3) We are then telling it to generate 32 filters and multiply across image.

Then over each epoch it will figure out filter gave the best signals to help match images to their labels After that the max-pooling image brings out enhanced features
```
model = models.Sequential()
# Convolutional Layer for each layer in the entire network
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32,32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
