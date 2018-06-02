import os

from PIL import Image


def resizeImage(base_dir, infile, output_dir="", size=(1024, 768), is_test=False):
    outfile = os.path.splitext(infile)[0] + "_resized"
    extension = os.path.splitext(infile)[1]

    if infile != outfile:
        try:
            im = Image.open(base_dir + "/" + infile)
            resized = im.resize(size, Image.ANTIALIAS)

            if not is_test:
                if not os.path.exists(output_dir + "/" + infile[:2]):
                    os.mkdir(output_dir + "/" + infile[:2])
                resized.save(output_dir + "/" + infile[:2] + "/" + outfile + extension, "JPEG")
            else:
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                resized.save(output_dir + "/" + outfile + extension, "JPEG")
            print(infile)
        except IOError:
            print("cannot reduce image for {}".format(infile))


if __name__ == "__main__":
    output_dir = "/resized"

    train_dir = "../data/train"
    train_dir_out = "../data/train/resized"
    val_dir = "../data/val"
    val_dir_out = "../data/val/resized"
    test_dir = "../data/test"
    test_dir_out = "../data/test/resized"

    if not os.path.exists(train_dir_out):
        os.mkdir(train_dir_out)

    if not os.path.exists(val_dir_out):
        os.mkdir(val_dir_out)

    if not os.path.exists(test_dir_out):
        os.mkdir(test_dir_out)

    for file in os.listdir(train_dir):
        resizeImage(train_dir, file, train_dir_out, (256, 256))

    for file in os.listdir(val_dir):
        resizeImage(val_dir, file, val_dir_out, (256, 256))

    for file in os.listdir(test_dir):
        resizeImage(test_dir, file, test_dir_out, (256, 256), True)
