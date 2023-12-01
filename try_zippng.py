import zipfile
from PIL import Image
import numpy as np


def store():
    a = np.uint8(255 * np.random.uniform(0, 0.01, (200, 200)))
    b = Image.fromarray(a)

    targ = zipfile.ZipFile("./mindata.zip", mode="w", compression=zipfile.ZIP_DEFLATED)
    with targ.open("data/dummy.png", mode="w") as f:
        b.save(f, format="png")
    targ.close()
    return a


def load():
    targ = zipfile.ZipFile("./mindata.zip", mode="r", compression=zipfile.ZIP_DEFLATED)
    with targ.open("data/dummy.png", mode="r") as f:
        img = Image.open(f)
        a = np.asarray(img)
    targ.close()
    return a


def main():
    orig = store()
    recon = load()
    print(np.sum(orig != recon))


if __name__ == "__main__":
    main()
