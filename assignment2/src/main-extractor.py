import csv
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np
from scipy import misc
from scipy.stats import kurtosis, skew
from skimage import img_as_float
from skimage.restoration import (denoise_bilateral, denoise_wavelet)

"""
Referencias

https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
https://www.kaggle.com/zeemeen/i-have-a-clue-what-i-am-doing-noise-patterns?scriptVersionId=2318696
http://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html

"""

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

    # image_path = "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/Motorola-Droid-Maxx/(MotoMax)1.jpg"

    image = None
    try:
        image = misc.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        exit(0)

    croped_image = img_as_float(crop_center(image, 512, 512))

    del image

    #denoised_image = denoise_bilateral(croped_image, sigma_color=0.05, sigma_spatial=15, multichannel=True)
    denoised_image = denoise_wavelet(croped_image, multichannel=True)
    noisy_image = np.array(croped_image - denoised_image)
    noisy_red, noisy_green, noisy_blue = np.array(noisy_image[:, :, 0]), \
                                         np.array(noisy_image[:, :, 1]), \
                                         np.array(noisy_image[:, :, 2])

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
        skew(noisy_blue.flatten()),

        # image class
        class_dict[image_class]
    ]

    print("Features da imagem {} calculadas!".format(procnum))
    return features


def main():
    pool = ThreadPoolExecutor(max_workers=8)
    features = []
    i = 0
    for folder_class in os.listdir("/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train"):
        print("Class {} is number {}".format(folder_class, class_dict[folder_class]))
        workers = []

        if i == 2:
            break

        for image_name in os.listdir(
                "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/{}/".format(folder_class)):
            image_path = "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/{}/{}".format(folder_class,
                                                                                                         image_name)

            td = pool.submit(worker, procnum=i, image_path=image_path, image_class=folder_class)
            workers.append(td)
            i += 1

        for worker_index in range(len(workers)):
            features.append(workers[worker_index].result())

        time.sleep(30)

    features = np.array(features)

    with open('features.csv', 'w') as csvfile:
        fieldnames = ['red_band_mean',
                      'red_band_var',
                      'red_band_std',
                      'red_band_kurtosis',
                      'red_band_skew',
                      'green_band_mean',
                      'green_band_var',
                      'green_band_std',
                      'green_band_kurtosis',
                      'green_band_skew',
                      'blue_band_mean',
                      'blue_band_var',
                      'blue_band_std',
                      'blue_band_kurtosis',
                      'blue_band_skew',
                      'image_class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for feature in features:
            writer.writerow({fieldnames[0]: feature[0],
                             fieldnames[1]: feature[1],
                             fieldnames[2]: feature[2],
                             fieldnames[3]: feature[3],
                             fieldnames[4]: feature[4],
                             fieldnames[5]: feature[5],
                             fieldnames[6]: feature[6],
                             fieldnames[7]: feature[7],
                             fieldnames[8]: feature[8],
                             fieldnames[9]: feature[9],
                             fieldnames[10]: feature[10],
                             fieldnames[11]: feature[11],
                             fieldnames[12]: feature[12],
                             fieldnames[13]: feature[13],
                             fieldnames[14]: feature[14],
                             fieldnames[15]: int(feature[15])})


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return np.array(img[starty:starty + cropy, startx:startx + cropx])


if __name__ == '__main__':
    main()
