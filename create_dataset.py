import numpy as np
import argparse
import sys
import os
import glob
import zipfile
import json
from PIL import Image


def get_img_and_pts(path, size):
    img = Image.open(path)
    with open(os.path.splitext(path)[0] + "_processed.json") as fp:
        marked = json.load(fp)

    result = dict()
    result["img"] = np.array(img)
    result["valid"] = np.array(marked["valid"], dtype=np.uint32)

    # if there are less invalid points than valid points,
    # randomly pick some points that are far away from all valid points
    fakes = marked["invalid"]

    shape = result["img"].shape
    lower_bounds = (size + 2, size + 2)
    upper_bounds = (shape[0] - (size + 2), shape[1] - (size + 2))

    target = len(result["valid"])
    count = len(fakes)

    pts = result["valid"]

    while count < target:
        f = np.random.randint(lower_bounds, upper_bounds, 2)
        dist = np.sqrt(np.sum((pts - f) ** 2, axis=1))

        if np.min(dist) > 3:
            count += 1
            fakes.append(f)

    result["invalid"] = np.array(fakes, dtype=np.uint32)

    return result


def save_patches(img, ID, pts, size, zfile, is_valid=True):
    l = size
    r = size + 1

    outs = []
    if is_valid:
        name = "data/img_%04d_valid_%06d.png"
    else:
        name = "data/img_%04d_invalid_%06d.png"

    for pt in pts:
        if (
            pt[0] < (l + 1)
            or pt[1] < (l + 1)
            or pt[0] > img.shape[0] - (r + 1)
            or pt[1] > img.shape[1] - (r + 1)
        ):
            continue
        ctx = img[pt[0] - l : pt[0] + r, pt[1] - l : pt[1] + r]
        outs.append(ctx)

    for i, x in enumerate(outs):
        # x[size - 5 : size + 6, size - 5 : size + 6] = [255, 0, 0]
        fname = name % (ID, i)
        z = Image.fromarray(x)
        with zfile.open(fname, "w") as f:
            z.save(f, format="png")


def show_context(ad, zfile, index=1, size=100):
    img = np.uint8(ad["img"] * 255)
    save_patches(img, index, ad["valid"], size, zfile, True)
    save_patches(img, index, ad["invalid"], size, zfile, False)


def runner(img_folder, zfile, size=13):
    psbs = glob.glob(os.path.join(img_folder, "*.png"))
    for i, iname in enumerate(psbs):
        print("processing", iname)
        data = get_img_and_pts(iname, size)
        show_context(data, zfile, i, size)


def main():
    parser = argparse.ArgumentParser("create-dataset")
    parser.add_argument(
        "-s", "--size", type=int, default=15, help="radius of each patch"
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        default="images",
        help="folder containing input images and JSONs",
    )
    parser.add_argument(
        "-o",
        "--output-zip",
        default="data",
        help="zip file with the output data",
    )

    a = parser.parse_args()
    zfile = zipfile.ZipFile(a.output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9)
    runner(a.input_folder, zfile, a.size)
    zfile.close()


if __name__ == "__main__":
    main()
