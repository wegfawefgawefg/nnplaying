'''
    putting images through nn's to get idea about what the architectures do to the images
'''

'''
TODO:
working on makeing the conv2d shit take in a 4 channel image
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable

import os
import time
import random
import numpy as np
from tqdm import tqdm
from matplotlib import image
from matplotlib import pyplot as plt
import logging
import termBar as tbar

# logger = logging.getLogger()
# old_level = logger.level
# logger.setLevel(100)

import architectures as arcs

def main():
    #   load image
    tbar.printSubHeader("Loading Image")
    fileName = 'im_20percent'
    ext = '.png'
    imRaw = image.imread(fileName + ext)
    im = torch.tensor(imRaw, dtype=torch.float32)

    #   image specs
    tbar.printSubHeader("Image Specs")
    numColorChannels = im.shape[2]
    print("numColorChannels: " + str(numColorChannels))
    print("imshape: " + str(im.shape))
    imWidth = im.shape[0]
    imFlat = im.flatten()
    flatLen = imFlat.shape[0]
    print("flatLen: " + str(flatLen))
    imWidth = im.shape[0]
    imHeight = im.shape[1]

    #   make net
    tbar.printSubHeader("Making Net")
    net = arcs.TinyConvNet(imWidth, imHeight, numColorChannels, flatLen)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #   train
    tbar.printSubHeader("Training Net")

    #   #   settings
    epochs = 100
    watch = True
    watchInterval = 10

    net.train()

    if watch:
        fig = plt.figure()
        data = np.zeros(im.shape)
        implt = plt.imshow(data)
        plt.show(block=False)
    
    inputIm = im.permute(2, 0, 1).view(1, numColorChannels, imWidth, imHeight)
    print("inputIm shape: " + str(inputIm.shape))

    pbar = tqdm(range(epochs))
    for i in pbar:
        barDescription = "Epoch %s" % (i)
        pbar.set_description(barDescription)

        optimizer.zero_grad()
        output = net(inputIm)

        if watch:
            if i % watchInterval == 0:
                outputImage = output.view(im.shape).detach().to("cpu").clamp(0.0, 1.0)
                implt.set_data(outputImage)
                fig.canvas.draw()
                time.sleep(0.01)

        loss = criterion(output, imFlat)
        loss.backward()
        optimizer.step()

    #   evaluate results
    tbar.printSubHeader("Eval Results")

    #   #   print nn convs
    conv1 = net.conv1.weight.data.numpy()
    print("conv1 shape: " + str(conv1.shape))
    
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    numConvs = net.numConvs
    columns = conv1.shape[0]
    rows = conv1.shape[1]
    for y in range(0, rows):
        for x in range(0, columns):
            pltConvIm = conv1[x][y]
            i = (y*columns) + x + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(pltConvIm)
    plt.show()
    quit()

    #   #   run through nn
    net.eval()
    output = net(inputIm)
    outputImage = output.view(im.shape).detach().numpy()
    outputImage = outputImage.clip(0.0, 1.0)
    displayImage = outputImage.copy()
    print(outputImage.dtype)
    print(outputImage.shape)

    outputImage = np.rollaxis(outputImage, 2, 0)
    newOutputImage = outputImage.copy()
    idx = [2, 0, 1]
    newOutputImage = newOutputImage[idx]
    print(outputImage.dtype)
    print(outputImage.shape)
    # outputImage = outputImage.transpose(3, 0, 1, 2)
    newOutputImage = np.rollaxis(newOutputImage, 0, 3)
    print(newOutputImage.dtype)
    print(newOutputImage.shape)

    #   save
    image.imsave(fileName + "_nn_out" + ext, displayImage)
    
    #   show
    plt.imshow(displayImage)
    plt.show()

if __name__ == '__main__':
    main()