import numpy as np
import skimage.io as skio
import skimage.transform as sktrans
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import imageio

from linear_NN import NN as MOD1
from convolutional_NN import CNN as MOD2
from recurrent_NN import RNN as MOD3
from recurrent_GRU_NN import RNN as MOD4
from recurrent_LSTM_NN import RNN as MOD5
from bidirectional_LSTM_NN import BidirectionalRNN as MOD6

class ModelMaker:
    def __init__(self, initfunc, wts_file, reshape=False):
        self.starter = initfunc
        self.net = initfunc()
        self.net.load_state_dict(torch.load(wts_file, map_location=torch.device("cpu")))
        self.net.eval()
        self.reshape = reshape
        #print(self.net)

    def __call__(self, image):
        x = torch.from_numpy(image)
        print(x.shape)
        # if there is an error it's likely due to shaping issues
        # add an unsqueeze(0) above if convolutional
        # don't if bidirectional
        if self.reshape:
            x = x.reshape(1, -1)
        else:
            x = x.unsqueeze(0)#.unsqueeze(0)
        res = self.net(x)
        res_max = torch.max(res)
        res_arg = torch.argmax(res)
        if res_arg == 0:
            res_max *= -1
        return res_max.detach().numpy()


MOD1.make = lambda: MOD1(input_size=225, num_classes=2)
MOD2.make = lambda: MOD2(input_size=1, num_classes=2)
MOD3.make = lambda: MOD3(15, 256, 2, 2, 15)
MOD4.make = lambda: MOD4(15, 256, 2, 2, 15)
MOD5.make = lambda: MOD5(15, 256, 2, 2, 15)
MOD6.make = lambda: MOD6(15, 256, 3, 2)

def run_on_image(maker, img_name, nn_type, num_epochs=None):
    img = skio.imread(img_name)
    tform = sktrans.EuclideanTransform(rotation=0, translation=(7, 7))
    padded_image = np.float32(
        sktrans.warp(
            img,
            tform.inverse,
            output_shape=(img.shape[0] + 14, img.shape[1] + 14),
            mode="reflect",
        )
    )

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    #plt.rcParams['figure.figsize'] = 220, 530
    axs[0].imshow(img, "Greys_r")

    res = np.zeros(img.shape, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #print(i,j)
            i2 = i + 7
            j2 = j + 7
            sub_image = padded_image[i2 - 7 : i2 + 8, j2 - 7 : j2 + 8]
            res[i, j] = maker(sub_image)

    axs[1].imshow(res, "Reds")

    plt.show()

    plt.imshow(res, "Reds", aspect='equal')
    name = Path(img_name).stem
    plt.axis("off")
    if (num_epochs):
        plt.savefig("../heatmaps/" + name + "_" + nn_type + "_epochs_"
                    + str(num_epochs) + ".png", bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig("../heatmaps/" + name + "_" + nn_type + ".png", bbox_inches='tight', pad_inches=0)

    #res = np.maximum(res, 0.95*np.max(res))
    res = res * 1.0 * (res > 0.85 * np.max((res)))

    plt.imshow(res, "Reds")
    plt.show()

    plt.imshow(img, "Greys")
    plt.imshow(res, "Reds", alpha=0.6)

    plt.show()

def main():
    wts_file = "../model_weights_recurrent"
    make1 = ModelMaker(MOD3.make, wts_file, False)
    a = np.zeros((15, 15), dtype=np.float32)
    #print(make1(a))
    f = "../002_07_L_01.png"
    #f = "../heatmaps/002_07_L_01_linear.png"
    run_on_image(make1, f, wts_file[17:])

if __name__ == "__main__":
    main()
