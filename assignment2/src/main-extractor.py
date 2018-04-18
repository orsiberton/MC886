import csv
import os


import numpy as np
from scipy import misc
from scipy.stats import kurtosis, skew
from skimage import img_as_float
from skimage.restoration import (denoise_wavelet)
from concurrent.futures import ThreadPoolExecutor
from mahotas.features.lbp import lbp

"""
Referencias

https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
https://www.kaggle.com/zeemeen/i-have-a-clue-what-i-am-doing-noise-patterns?scriptVersionId=2318696
http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html

"""
NUMBER_OF_FEATURES = 195
class_dict = {
    'Motorola-Droid-Maxx': 0,
    'iPhone-4s': 1,
    'LG-Nexus-5x': 2,
    'Motorola-Nexus-6': 3,
    'iPhone-6': 4,
    'Sony-NEX-7': 5,
    'Samsung-Galaxy-Note3': 6,
    'HTC-1-M7': 7,
    'Samsung-Galaxy-S4': 8,
    'Motorola-X': 9
}


def worker(procnum, image_path, image_class):
    print("Calculando features da imagem {}".format(procnum))

    #image_path = "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/Motorola-Droid-Maxx/(MotoMax)1.jpg"

    image = None
    try:
        image = misc.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        exit(0)

    croped_image = img_as_float(crop_center(image, 512, 512))

    del image

    # denoised_image = denoise_bilateral(croped_image, sigma_color=0.05, sigma_spatial=15, multichannel=True)
    denoised_image = denoise_wavelet(croped_image, multichannel=True)
    noisy_image = np.array(croped_image - denoised_image)
    noisy_red, noisy_green, noisy_blue = np.array(noisy_image[:, :, 0]), \
                                         np.array(noisy_image[:, :, 1]), \
                                         np.array(noisy_image[:, :, 2])

    lbp_red_features = lbp(noisy_red, 2, 9)
    lbp_green_features = lbp(noisy_green, 2, 9)
    lbp_blue_features = lbp(noisy_blue, 2, 9)

    features = [
        # red band
        noisy_red.mean(),
        np.var(noisy_red),
        noisy_red.std(),
        kurtosis(noisy_red.flatten()),
        skew(noisy_red.flatten()),

        # green band
        noisy_green.mean(),
        np.var(noisy_green),
        noisy_green.std(),
        kurtosis(noisy_green.flatten()),
        skew(noisy_green.flatten()),

        # blue band
        noisy_blue.mean(),
        np.var(noisy_blue),
        noisy_red.std(),
        kurtosis(noisy_blue.flatten()),
        skew(noisy_blue.flatten())
    ]

    for feature in lbp_red_features:
        features.append(feature)

    for feature in lbp_green_features:
        features.append(feature)

    for feature in lbp_blue_features:
        features.append(feature)

    # image class
    features.append(class_dict[image_class])

    print("Features da imagem {} calculadas!".format(procnum))
    return features


def main():
    pool = ThreadPoolExecutor(max_workers=8)
    features = []
    i = 0
    for folder_class in os.listdir("/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train"):
        print("Class {} is number {}".format(folder_class, class_dict[folder_class]))
        workers = []

        # if i >= 100:
        #     break

        for image_name in os.listdir(
                "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/{}/".format(folder_class)):
            image_path = "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/{}/{}".format(folder_class,
                                                                                                         image_name)

            td = pool.submit(worker, procnum=i, image_path=image_path, image_class=folder_class)
            workers.append(td)
            i += 1
            #break

        for worker_index in range(len(workers)):
            features.append(workers[worker_index].result())

        #break

    features = np.array(features)

    with open('features.csv', 'w') as csvfile:
        fieldnames = ["f{}".format(i) for i in range(1, NUMBER_OF_FEATURES + 1)]
        fieldnames.append('image_class')

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for feature in features:
            feature_dict = {}
            for index, field_name in enumerate(fieldnames):
                if field_name != 'image_class':
                    feature_dict[field_name] = feature[index]
                else:
                    feature_dict[field_name] = int(feature[index])

            writer.writerow(feature_dict)


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return np.array(img[starty:starty + cropy, startx:startx + cropx])


if __name__ == '__main__':
    main()
