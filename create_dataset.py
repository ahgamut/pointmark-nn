import psd_tools
import numpy as np
import argparse
import sys
import os
import glob
import zipfile
from PIL import Image

ShapeLayer = psd_tools.api.layers.ShapeLayer
PixelLayer = psd_tools.api.layers.PixelLayer


def get_img_and_pts(path):
    psd = psd_tools.PSDImage.open(path)

    result = dict()
    base = (0, 0)
    pts = []
    for layer in psd:
        if layer.name == "Layer 0" or type(layer) == PixelLayer:
            if result.get("base") is not None:
                raise RuntimeError(
                    "multiple PixelLayers in the PSB, which one is the image?"
                )
            result["img"] = layer.numpy(channel="color")
            base = (np.abs(layer.bbox[1]), np.abs(layer.bbox[0]))
            result["bbox"] = layer.bbox
            result["base"] = base
            continue
        elif "Ellipse" in layer.name or type(Layer) == ShapeLayer:
            pt = [
                0.5 * (layer.bbox[1] + layer.bbox[3]),
                0.5 * (layer.bbox[2] + layer.bbox[0]),
            ]
            pts.append(pt)
        elif False and layer.name == "Corners":
            print(layer.__dict__)
            a = layer.numpy()
            r, c = np.where(a[:, :, 3] != 0)
            pts = np.column_stack([r, c]) + (layer.bbox[1], layer.bbox[0]) + base
            pts = _split(pts, 25)
            print(len(pts))
            result["real"] = pts

    pts = np.array(pts, dtype=np.int32) + base
    result["real"] = pts
    return result


def load_fakes(ad, size=100):
    shape = ad["img"].shape
    lower_bounds = (size + 2, size + 2)
    upper_bounds = (shape[0] - (size + 2), shape[1] - (size + 2))

    fakes = []
    target = len(ad["real"])
    count = 0

    pts = ad["real"]

    while count < target:
        f = np.random.randint(lower_bounds, upper_bounds, 2)
        dist = np.sqrt(np.sum((pts - f) ** 2, axis=1))

        if np.min(dist) > size / 2:
            count += 1
            fakes.append(f)

    ad["fake"] = np.array(fakes, dtype=np.uint32)


def save_patches(img, ID, pts, size, zfile, is_real=True):
    l = size
    r = size + 1

    outs = []
    if is_real:
        name = "data/img_%04d_real_%06d.png"
    else:
        name = "data/img_%04d_fake_%06d.png"

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
    load_fakes(ad, size)
    save_patches(img, index, ad["real"], size, zfile, True)
    save_patches(img, index, ad["fake"], size, zfile, False)


def runner(img_folder, zfile, size=100):
    psbs = glob.glob(os.path.join(img_folder, "*.psb"))
    for i, iname in enumerate(psbs):
        print("processing", iname)
        data = get_img_and_pts(iname)
        show_context(data, zfile, i, size)


def main():
    parser = argparse.ArgumentParser("create-dataset")
    parser.add_argument(
        "-s", "--size", type=int, default=100, help="radius of each patch"
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        default="images",
        help="folder containing input PSB files",
    )
    parser.add_argument(
        "-o",
        "--output-zip",
        default="example.zip",
        help="zip file with the output data",
    )

    a = parser.parse_args()
    zfile = zipfile.ZipFile(a.output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9)
    runner(a.input_folder, zfile, a.size)
    zfile.close()


if __name__ == "__main__":
    main()
