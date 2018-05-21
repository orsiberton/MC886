from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from keras import layers

img_width, img_height = 256, 256

# Returns a compiled model identical to the previous one
model = load_model('small_last4.h5')

train_dir = "../data/train/resized"
val_dir = "../data/val/resized"
test_dir = "../data/test/resized"

validation_datagen = ImageDataGenerator(rescale=1./255)
val_batchsize = 10

validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

results = model.evaluate_generator(
    validation_generator,
    steps=validation_generator.samples/validation_generator.batch_size,
    workers=4,
    use_multiprocessing=True)

predictions = model.predict_generator(
    validation_generator,
    steps=validation_generator.samples/validation_generator.batch_size,
    workers=4,
    use_multiprocessing=True)

print model.metrics_names
print results

print "Predictions"
print predictions
