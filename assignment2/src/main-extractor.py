import csv
import multiprocessing
import os
import time

import numpy as np
from scipy import misc
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, skew
from skimage.restoration import denoise_wavelet

"""
Referencias

https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
https://www.kaggle.com/zeemeen/i-have-a-clue-what-i-am-doing-noise-patterns?scriptVersionId=2318696

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


def worker(procnum, send_end, image_path, image_class):
    print("Calculando features da imagem {}".format(procnum))

    image = None
    try:
        image = misc.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        exit(0)

    croped_image = crop_center(image, 512, 512)

    gaussian_image = gaussian_filter(croped_image, sigma=5)
    denoised_image = denoise_wavelet(croped_image, wavelet='db8', multichannel=True, convert2ycbcr=True)

    image_minus_gaussian = croped_image - gaussian_image
    image_minus_denoised = croped_image - denoised_image

    # TODO do this for each color band and use wavelets for each color band too
    features = [
        # gaussian features
        np.var(image_minus_gaussian),
        kurtosis(image_minus_gaussian).mean(),
        skew(image_minus_gaussian).mean(),

        # denoised features
        np.var(image_minus_denoised),
        kurtosis(image_minus_denoised).mean(),
        skew(image_minus_denoised).mean(),

        # image class
        class_dict[image_class]
    ]

    print(features)
    send_end.send(features)


def main():
    features = []
    i = 0
    for folder_class in os.listdir("/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train"):
        print("Class {} is number {}".format(folder_class, class_dict[folder_class]))
        jobs = []
        pipe_list = []
        for image_name in os.listdir(
                "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/{}/".format(folder_class)):
            image_path = "/home/CIT/bberton/Documents/unicamp/MC886/assignment2/data/train/{}/{}".format(folder_class,
                                                                                                         image_name)

            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=worker, args=(i, send_end, image_path, folder_class))
            jobs.append(p)
            pipe_list.append(recv_end)
            p.start()
            i += 1

        for proc in jobs:
            proc.join()

        for result in pipe_list:
            features.append(result.recv())

        time.sleep(120)

    features = np.array(features)

    print(features)

    with open('features.csv', 'w') as csvfile:
        fieldnames = ['gaussian_variance',
                      'gaussian_kurtosis',
                      'gaussian_skew',
                      'denoised_variance',
                      'denoised_kurtoses',
                      'denoised_skew',
                      'image_class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for feature in features:
            writer.writerow({'gaussian_variance': feature[0],
                             'gaussian_kurtosis': feature[1],
                             'gaussian_skew': feature[2],
                             'denoised_variance': feature[3],
                             'denoised_kurtoses': feature[4],
                             'denoised_skew': feature[5],
                             'image_class': int(feature[6])})


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


if __name__ == '__main__':
    main()
