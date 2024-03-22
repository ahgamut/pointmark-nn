import numpy as np
import skimage.io as skio
import skimage.transform as sktrans
import matplotlib.pyplot as plt
import torch

from linear_NN import NN as MOD1
from convolutional_NN import CNN as MOD2
from bidirectional_LSTM_NN import BidirectionalRNN as MOD6
"""
from recurrent_NN import RNN as MOD3
from recurrent_GRU_NN import RNN as MOD4
from recurrent_LSTM_NN import RNN as MOD5
"""


class ModelMaker:
    def __init__(self, initfunc, wts_file, reshape=False):
        self.starter = initfunc
        self.net = initfunc()
        self.net.load_state_dict(torch.load(wts_file, map_location=torch.device("cpu")))
        self.net.eval()
        self.reshape = reshape
        print(self.net)

    def __call__(self, image):
        x = torch.from_numpy(image)
        print(x.shape)
        # if there is an error it's likely due to shaping issues
        # add an unsqueeze(0) above if convolutional
        # don't if bidirectional
        if self.reshape:
            x = x.reshape(1, -1)
        else:
            x = x.unsqueeze(0)
        res = self.net(x)
        res_max = torch.max(res)
        res_arg = torch.argmax(res)
        if res_arg == 0:
            res_max *= -1
        return res_max.detach().numpy()


MOD1.make = lambda: MOD1(input_size=625, num_classes=2)
MOD2.make = lambda: MOD2(input_size=1, num_classes=2)
MOD6.make = lambda: MOD6(25, 256, 3, 2)
"""
MOD3.make = lambda: MOD3(25, 256, 2, 2)
MOD4.make = lambda: MOD4(25, 256, 2, 2)
MOD5.make = lambda: MOD5(25, 256, 2, 2)
"""


def run_on_image(maker, img):
    tform = sktrans.EuclideanTransform(rotation=0, translation=(12, 12))
    padded_image = np.float32(
        sktrans.warp(
            img,
            tform.inverse,
            output_shape=(img.shape[0] + 24, img.shape[1] + 24),
            mode="reflect",
        )
    )

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(img, "Greys_r")

    res = np.zeros(img.shape, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            print(i,j)
            i2 = i + 12
            j2 = j + 12
            sub_image = padded_image[i2 - 12 : i2 + 13, j2 - 12 : j2 + 13]
            res[i, j] = maker(sub_image)

    axs[1].imshow(res, "Reds")
    plt.show()


def main():
    make1 = ModelMaker(MOD6.make, "../model_weights_bidirectional", False)
    a = np.zeros((25, 25), dtype=np.float32)
    print(make1(a))
    image = skio.imread("../examples.png")
    print(image.shape)
    run_on_image(make1, image)


if __name__ == "__main__":
    main()
