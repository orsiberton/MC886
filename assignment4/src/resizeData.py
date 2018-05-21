from PIL import Image
import os, sys

def resizeImage(base_dir, infile, output_dir="", size=(1024,768)):
     outfile = os.path.splitext(infile)[0]+"_resized"
     extension = os.path.splitext(infile)[1]

     if (cmp(extension, ".jpg")):
        return

     if infile != outfile:
        try :
            im = Image.open(base_dir +"/"+ infile)
            resized = im.resize(size, Image.ANTIALIAS)
            if not os.path.exists(output_dir+"/"+infile[:2]):
                os.mkdir(output_dir+"/"+infile[:2])
            resized.save(output_dir+"/"+infile[:2]+"/"+outfile+extension,"JPEG")
            print infile
        except IOError:
            print "cannot reduce image for ", infile


if __name__=="__main__":
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
        resizeImage(test_dir, file, test_dir_out, (256, 256))
