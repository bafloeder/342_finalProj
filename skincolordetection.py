import os
from scipy import misc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


np.set_printoptions(threshold=np.nan)

def load_image(filename):
    path = './'
    # image = misc.imread(os.path.join(path, filename), flatten=0)
    image = Image.open(filename)
    return image

def histogram(image):
    data = image.getdata()
    width, height = image.size
    hist = {}
    for i in range(height):
        for j in range(width):
            pixel = data.getpixel((j,i))
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            x, y, z = (math.floor(r / 255. * 10), math.floor(g / 255. * 10), math.floor(b / 255. * 10))
            s = r + g + b
            x = math.floor(r / float(s) * 100)
            y = math.floor(g / float(s) * 100)
            z = math.floor(b / float(s) * 100)

            if (x, y) in hist:
                hist[(x, y)] += 1
            else:
                hist[(x, y)] = 1
            # avg = int((pixel[0] + pixel[1] + pixel[2])/3)
            # hist[avg] = hist[avg] + 1
    # plt.hist(hist, 256)
    # plt.show()
    return hist

def train_img(images):
    hist = 1

def segmentation(image):
    training_images = []
    threshold =0.00000001
    # print(image.size)
    width, height = image.size
    data = image.getdata()
    for i in range(1,15):
        training_images.append('skin'+str(i)+'.jpg')

    hsvdict = {}
    for path in training_images:
        training_image = Image.open(path)
        training_image_data = training_image.getdata()
        hist = histogram(training_image)
        hsvdict.update(hist)
        # print(hsvdict)
    sum = 0
    for val in hsvdict.values():
        sum += val
    for j in hsvdict.keys():
        hsvdict[j] /= float(sum)

    # print(hsvdict)

    result = [[0 for x in range(width)] for x in range(height)]

    for i in range(height):
        for j in range(width):
            pixel = data.getpixel((j,i))
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            x, y, z = (math.floor(r / 255. * 10), math.floor(g / 255. * 10), math.floor(b / 255. * 10))

            s = r + g + b
            if (s != 0):
                x = math.floor(r / float(s) * 100)
                y = math.floor(g / float(s) * 100)
                z = math.floor(b / float(s) * 100)

            if (x, y) in hsvdict and hsvdict[(x,y)] > threshold:
                # print(x,y)
                result[i][j] = (r,g,b)
                # print(i,j)
            else:
                result[i][j] = (0, 0, 0)
                # print(i,j)

    return result


# img = load_image('joy1.bmp')
# hist = histogram(img)
# threshold = 0.000001
# new_img = segmentation(img)
# new_img = (np.array(new_img)).astype(np.uint8)
# result = Image.fromarray(new_img)
# result.save('joy1_2.bmp', 'bmp')
