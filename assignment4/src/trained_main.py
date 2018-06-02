import os

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

img_width, img_height = 256, 256

# Returns a compiled model identical to the previous one
model = load_model('inception_v3_pool_nodrop_soft.h5')

train_dir = "../data/train/resized"
val_dir = "../data/val/resized"
test_dir = "../data/test/resized"

validation_datagen = ImageDataGenerator(rescale=1. / 255)
val_batchsize = 10

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

results = model.evaluate_generator(
    validation_generator,
    steps=validation_generator.samples / validation_generator.batch_size,
    workers=4,
    use_multiprocessing=True)

predictions = model.predict_generator(
    validation_generator,
    steps=validation_generator.samples / validation_generator.batch_size,
    workers=4,
    use_multiprocessing=True)

print(model.metrics_names)
print(results)

print("Predictions")
print(predictions)

y_pred = [np.argmax(pred) for pred in predictions]

class_names = []
for i in range(83):
    class_names.append("Class {0:02d}".format(i))

y_true = []

for file in sorted(os.listdir("../data/val")):
    if file[-4:] == ".jpg":
        y_true.append(int(file[0:2]))

print(classification_report(y_true, y_pred, target_names=class_names))
acc = accuracy_score(y_true, y_pred, normalize=True) * 100

print(acc)

prob = (1 / 83) * 100
norm_acc = ((acc - prob) / (100 - prob)) * 100

print(norm_acc)
