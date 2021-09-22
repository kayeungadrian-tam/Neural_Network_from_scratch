from numpy.lib.arraypad import pad
import pandas as pd
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

file_path = "../input"
files = os.listdir(file_path)
dataset = pd.read_csv(os.path.join(file_path, "train.csv"))

train_x = dataset.drop("label", axis=1)
test_row = train_x.iloc[42]
test_digit = test_row.values.reshape(28,28)
img = test_digit

image = img.reshape(28, 28, 1)
shape = image.shape



kshape = (3, 3)
stride = (1, 1)
filters = 5

p = 0

bias = np.random.randn(filters, 1)
weights = np.random.randn(3, 3, 1, filters)


init = np.zeros((26, 26))

out = np.zeros(())

fig, axs = plt.subplots(1,2)
axs = axs.ravel()






input_shape = img.shape

#########################


output_shape = (int((input_shape[0] - kshape[0] + 2 * p) / stride[0]) + 1, 
                    int((input_shape[1] - kshape[1] + 2 * p) / stride[1]) + 1, filters)

#########################

out = np.zeros(output_shape)

print(out.shape)

for f in range(filters):
    zeros_h = np.zeros((shape[1], shape[2])).reshape(-1, shape[1], shape[2])
    zeros_v = np.zeros((shape[0]+2, shape[2])).reshape(shape[0]+2, -1, shape[2])
    padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
    padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols

    print(padded_img.shape)

    col = 0
    row = 0

    rv = 0
    cimg = []
    for r in range(kshape[0], shape[0]+1, stride[0]):
        cv = 0
        for c in range(kshape[1], shape[1]+1, stride[1]):
            chunk = padded_img[rv:r, cv:c]
            soma = (np.multiply(chunk, weights[:, :, :, f]))
            summa = soma.sum() + np.random.uniform()
            
            # axs[0].imshow(padded_img.squeeze(axis=2))
            # rect = patches.Rectangle((col%26, row), 3, 3, linewidth=3, edgecolor="r", facecolor="none")
            # axs[0].add_patch(rect)
            
            # init[row,col%26] = summa
            # ax1 = axs[1].imshow(init)
            # plt.imshow(init, cmap="jet")
            # fig.colorbar(ax1)
            
            # plt.pause(0.1)
            # axs[0].clear()
            # axs[1].clear()
            
            
            col += 1
            cimg.append(summa)
            cv+=stride[1]
        row += 1
        rv+=stride[0]
    cimg = np.array(cimg).reshape(int(rv/stride[0]), int(cv/stride[1]))
    out[:, :, f] = cimg
    
    
