from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.applications import vgg16



def PredictImg(modelStructure,modelWeights,TestImageSet):
    f = Path(modelStructure)
    model_structure = f.read_text()

    # Recreate the Keras model object from the json data
    model = model_from_json(model_structure)

    # Re-load the model's trained weights
    model.load_weights(modelWeights)

    # Load an image file to test, resizing it to 64x64 pixels (as required by this model)
    img = image.load_img(TestImageSet, target_size=(224, 224))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
    images = np.expand_dims(image_array, axis=0)

    # Normalize the data
    images = vgg16.preprocess_input(images)

    # Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
    feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = feature_extraction_model.predict(images)

    # Given the extracted features, make a final prediction using our own model
    results = model.predict(features)
    print(results)
    if results[0][0] == 1:
        print("Bottels")
    elif results[0][1] == 1:
        print("Cups")
    else:
        print("Coffeemugs")

PredictImg(modelStructure = "model_structure.json",modelWeights = "model_weights.h5",TestImageSet ="CupsClassifier.jpg")


# Load the json file that contains the model's structure

