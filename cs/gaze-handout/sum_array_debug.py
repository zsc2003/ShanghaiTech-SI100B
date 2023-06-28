import pandas as pd
import numpy as np
from gazelib.utils import decode_base64_img
import copy
#import pysnooper
from gazelib.conv2d import gaussian_knl, sobel_hrztl, sobel_vtcl


def conv2d3x3(im: np.ndarray, kernel: np.ndarray):
    assert im.shape[0] >= 3
    assert im.shape[1] >= 3

    reta = np.zeros((im.shape[0] - 2, im.shape[1] - 2), dtype=np.float64)
    retb = np.zeros((im.shape[0] - 2, im.shape[1] - 2), dtype=np.float64)
    retc = np.zeros((im.shape[0] - 2, im.shape[1] - 2), dtype=np.float64)
    retd = np.zeros((im.shape[0] - 2, im.shape[1] - 2), dtype=np.float64)
    n=im.shape[0]
    m=im.shape[1]#F n*m
    for i in range(n-2):
        for j in range(m-2):
            temp = im[i: i + 3, j: j + 3] * kernel
            a = temp[0, 0] + temp[0, 1] + temp[0, 2] +\
                temp[1, 0] + temp[1, 1] + temp[1, 2] +\
                temp[2, 0] + temp[2, 1] + temp[2, 2]
            b = sum(temp[0] + temp[1] + temp[2])
            c = sum(sum(temp))

            reta[i][j] = a
            retb[i][j] = b
            retc[i][j] = c

            #print("a:",a);print("b:",b);print("c",c)
    #print(ret)
    print(reta);print(retb);print(retc)
    assert ret.shape[0] == im.shape[0] - 2
    assert ret.shape[1] == im.shape[1] - 2

    return ret


if __name__ == '__main__':
    from gazelib.utils import *
    import numpy as np
    import time
    import matplotlib.pyplot as plt
    import random

    train_df = load_train_csv_as_df()

    # transform data into numpy arrays
    def df2nptensor(df):
        imgs = []
        imgs_HOG = []
        gaze_dirs = []

        print_interval = 1000
        print_cnter = 0

        for _, i in df.iterrows():
            if print_cnter % print_interval == 0:
                print("[{} / {}]".format(print_cnter, len(df)), end='\r')
            print_cnter += 1
            im_arr = decode_base64_img(i['image_base64'])
            gaze_dirs.append([i['yaw'], i['pitch']])
            im = im_arr / 255

            imgs.append(im)

        gaze_dirs = np.array(gaze_dirs)
        imgs = np.array(imgs)

        return gaze_dirs, imgs


    # For effciency, we only takes first 5,000 samples. Pick subject 5 as validation
    # set and the rest of the dataset as training set
    SAMPLE_NUM = 5000
    print("Start to generate sampled dataset, it may take ~10s.")
    train_Y, train_X = df2nptensor(train_df[train_df["subject_id"] != 5][: int(SAMPLE_NUM * 0.8)])
    val_Y, val_X = df2nptensor(train_df[train_df["subject_id"] == 5][: int(SAMPLE_NUM * 0.2)])


    print("train_X.shape: {}".format(train_X.shape))
    print("train_Y.shape: {}".format(train_Y.shape))
    print("val_X.shape: {}".format(val_X.shape))
    print("val_X.shape: {}".format(val_Y.shape))


    # im = val_X[10]
    # kernel = gaussian_knl

    im = np.array([
        [3.5, 3.5, 2.5, 1.5, 0.5],
        [0.5, 0.5, 1.5, 3.5, 1.5],
        [3.5, 1.5, 2.5, 2.5, 3.5],
        [2.5, 0.5, 0.5, 2.5, 2.5],
        [2.0, 0.5, 0.5, 0.5, 1.5]
    ])
    kernel = np.array([
        [0.5, 1.5, 2.5],
        [2.5, 2.5, 0.5],
        [0.5, 1.5, 2.5]
    ])
    ret = conv2d3x3(im, kernel)