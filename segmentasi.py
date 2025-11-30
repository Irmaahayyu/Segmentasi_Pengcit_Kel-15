import os
import numpy as np
from skimage import io, color, img_as_float
from scipy import ndimage as ndi
from PIL import Image

# ===== LOAD =====
def load_gray(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return img_as_float(img)

def normalize01(x):
    x = x - x.min()
    mx = x.max()
    if mx == 0:
        return x
    return x / mx

def contrast_stretch(img, low=2, high=98):
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi - lo == 0:
        return img
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0, 1)

def save_img(arr, path):
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)

# ===== ROBERTS =====
def op_roberts(img):
    Kx = np.array([[1, 0],
                   [0,-1]], float)
    Ky = np.array([[0, 1],
                   [-1, 0]], float)
    gx = ndi.convolve(img, Kx, mode="reflect")
    gy = ndi.convolve(img, Ky, mode="reflect")
    mag = np.hypot(gx, gy)
    mag = normalize01(mag)
    mag = mag ** 1.8
    mag = mag * 0.8
    return mag

# ===== PREWITT (DISETUP SEBAGAI 'SOBEL' RELIEF ABU TERANG) =====
def op_prewitt(img):
    Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], float)
    Ky = np.array([[ 1,  1,  1],
                   [ 0,  0,  0],
                   [-1, -1, -1]], float)
    gx = ndi.convolve(img, Kx, mode="reflect")
    gy = ndi.convolve(img, Ky, mode="reflect")
    mag = np.hypot(gx, gy)
    mag = normalize01(mag)
    mag = contrast_stretch(mag, 2, 99)
    mag = mag ** 0.8
    return mag

# ===== SOBEL (DISETUP SEBAGAI 'PREWITT' GARIS PUTIH TEGAS) =====
def op_sobel(img):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], float)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], float)
    gx = ndi.convolve(img, Kx, mode="reflect")
    gy = ndi.convolve(img, Ky, mode="reflect")
    mag = np.hypot(gx, gy)
    mag = normalize01(mag)
    mag = contrast_stretch(mag, 1, 98)
    mag = mag ** 0.6
    return mag

# ===== FREI-CHEN =====
def op_freichen(image):
    s2 = np.sqrt(2)
    Gx = np.array([[ 1,  s2,  1],
                   [ 0,  0,  0],
                   [-1, -s2, -1]], float)
    Gy = np.array([[ 1,   0, -1],
                   [ s2,  0, -s2],
                   [ 1,   0, -1]], float)
    fx = ndi.convolve(image, Gx, mode="reflect")
    fy = ndi.convolve(image, Gy, mode="reflect")
    mag = np.hypot(fx, fy)
    mag = normalize01(mag)
    mag = contrast_stretch(mag, 1, 95)
    mag = mag ** 1.1
    mag = mag * 0.9
    return mag

# ===== DATA =====
images = {
    "original": "images/original.jpg",
    "grayscale": "images/grayscale.jpg",
    "saltpepper": "images/saltpepper.jpg",
    "gaussian": "images/gaussian.jpg"
}

os.makedirs("hasil_segmentasi", exist_ok=True)

methods = {
    "Roberts": op_roberts,
    "Prewitt": op_prewitt,
    "Sobel": op_sobel,
    "FreiChen": op_freichen
}

# ===== PROSES =====
for name, path in images.items():
    im = load_gray(path)
    for mname, func in methods.items():
        edge = func(im)
        save_img(edge, f"hasil_segmentasi/{name}_{mname}.png")

print("=== SELESAI ===")
print("Hasil tersimpan di folder: hasil_segmentasi/")
