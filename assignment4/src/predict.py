import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

target_size = (256, 256)  # fixed size for InceptionV3 architecture


def predict(model, img, target_size):
    """Run model prediction on image
    Args:
      model: keras model
      img: PIL format image
      target_size: (w,h) tuple
    Returns:
      list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    hue = np.array([x, x, x, x]).reshape((4, 256, 256, 3))
    print(model.predict_classes(hue, batch_size=10))

    return preds[0]


def plot_preds(image, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
      image: PIL image
      preds: list of predicted labels and their probabilities
    """
    plt.imshow(image)
    plt.axis('off')

    plt.figure()
    labels = ("cat", "dog")
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    # a.add_argument("--image", help="path to image")
    args = a.parse_args()

    print("hue")
    model = load_model("inception_v3_flat_relu_soft.h5")
    print("hue2")
    img = Image.open("renato.jpeg")
    preds = predict(model, img, target_size)
    print(preds)
    print(str(np.argmax(preds)))
    # plot_preds(img, preds)
