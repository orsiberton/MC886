import csv
import os

import numpy as np
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import classification_report, accuracy_score

images = {}
csv_file = csv.DictReader(open("../data/MO444_dogs_test.csv"))
for csv_line in csv_file:
    images[csv_line['image_name']] = int(csv_line['class'])

print(images)

img_width, img_height = 256, 256

# Returns a compiled model identical to the previous one
model = load_model('inception_v3_pool_nodrop_soft.h5')

test_dir = "../data/test/resized"
# test_dir = "../data/berton/resized"

x = []
for file in sorted(os.listdir(test_dir)):
    img = Image.open(test_dir + "/" + file)
    img_input = image.img_to_array(img)
    img_input = np.expand_dims(img_input, axis=0)
    img_input = preprocess_input(img_input)
    x.append(img_input)

y_pred = []
for x_input in x:
    y_pred.append(np.argmax(model.predict(x_input)))

# x = np.array(x).reshape((len(x), 256, 256, 3))
# y_pred = model.predict_classes(x, batch_size=10)

print(y_pred)

class_names = []
for i in range(83):
    class_names.append("Class {0:02d}".format(i))

y_true = []

for file in sorted(os.listdir("../data/test")):
    if file[-4:] == ".jpg":
        y_true.append(images[file])

print(classification_report(y_true, y_pred, target_names=class_names))
acc = accuracy_score(y_true, y_pred, normalize=True) * 100

print(acc)

prob = (1 / 83) * 100
norm_acc = ((acc - prob) / (100 - prob)) * 100

print(norm_acc)
