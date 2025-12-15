import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
from scipy import ndimage as ndi
from PIL import Image

# ================== UTILITAS ==================
def load_gray_8bit(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = (img * 255).astype(np.float64)
    return img

def save_img_8bit(img, path):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def match_size(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]

# ================== MSE ==================
def mse_8bit(a, b):
    if a.shape != b.shape:
        return -1
    err = np.sum((a.astype(float) - b.astype(float)) ** 2)
    err /= float(a.shape[0] * a.shape[1])
    return err

# ================== OPERATOR DETEKSI TEPI ==================
def roberts(img):
    Gx = np.array([[1, 0],
                   [0, -1]])
    Gy = np.array([[0, -1],
                   [1, 0]])
    gx = ndi.convolve(img, Gx, mode="reflect")
    gy = ndi.convolve(img, Gy, mode="reflect")
    return np.sqrt(gx**2 + gy**2)

def prewitt(img):
    Gx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
    Gy = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])
    gx = ndi.convolve(img, Gx, mode="reflect")
    gy = ndi.convolve(img, Gy, mode="reflect")
    return np.sqrt(gx**2 + gy**2)

def sobel(img):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    gx = ndi.convolve(img, Gx, mode="reflect")
    gy = ndi.convolve(img, Gy, mode="reflect")
    return np.sqrt(gx**2 + gy**2)

def freichen(img):
    s2 = np.sqrt(2)
    Gx = np.array([[-1, 0, 1],
                   [-s2, 0, s2],
                   [-1, 0, 1]])
    Gy = np.array([[1, s2, 1],
                   [0, 0, 0],
                   [-1, -s2, -1]])
    gx = ndi.convolve(img, Gx, mode="reflect")
    gy = ndi.convolve(img, Gy, mode="reflect")
    return np.sqrt(gx**2 + gy**2)

methods = {
    "Roberts": roberts,
    "Prewitt": prewitt,
    "Sobel": sobel,
    "Frei-Chen": freichen
}

# ================== DATASET ==================
datasets = {
    "portrait": "Images/Potrait",
    "landscape": "Images/Landscape"
}

image_types = {
    "portrait": [
        "original",
        "grayscale",
        "gaussian",
        "saltpepper",
        "gray_ga_1_median",
        "gray_sp_1_median",
        "gray_sp_2_median",
        "rgb_ga_1_median",
        "rgb_ga_2_median",
        "rgb_sp_1_median",
        "rgb_sp_2_median"
    ],
    "landscape": [
        "original",
        "grayscale",
        "gaussian",
        "saltpepper",
        "gray_ga_1_median",
        "gray_ga_2_median",
        "gray_sp_1_median",
        "gray_sp_2_median",
        "rgb_ga_1_median",
        "rgb_ga_2_median",
        "rgb_sp_1_median",
        "rgb_sp_2_median"
    ]
}

os.makedirs("hasil", exist_ok=True)
hasil_mse = []

# ================== PROSES ==================
for ds, folder in datasets.items():
    os.makedirs(f"hasil/{ds}", exist_ok=True)

    ref = load_gray_8bit(f"{folder}/original.jpg")

    for img_name in image_types[ds]:
        img = load_gray_8bit(f"{folder}/{img_name}.jpg")

        for mname, func in methods.items():
            edge = func(img)
            save_img_8bit(edge, f"hasil/{ds}/{img_name}_{mname}.png")

            ref_c, edge_c = match_size(ref, edge)

            hasil_mse.append({
                "Dataset": ds,
                "Citra": img_name,
                "Metode": mname,
                "MSE": mse_8bit(ref_c, edge_c)
            })

# ================== TABEL PERBANDINGAN MSE ==================
df = pd.DataFrame(hasil_mse)

df_noise = df[df["Citra"].isin(["gaussian", "saltpepper"])]

tabel_mse = df_noise.pivot_table(
    values="MSE",
    index="Metode",
    columns="Citra",
    aggfunc="mean"
)

print("\n=========== TABEL PERBANDINGAN MSE (GAUSSIAN vs SALT PEPPER) ===========\n")
print(tabel_mse.round(2))

# ================== GRAFIK PERBANDINGAN MSE ==================
labels = tabel_mse.index.tolist()
gaussian = tabel_mse["gaussian"].values
saltpepper = tabel_mse["saltpepper"].values

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(9, 5))

bars1 = plt.bar(x - width/2, gaussian, width, label="Gaussian Noise")
bars2 = plt.bar(x + width/2, saltpepper, width, label="Salt & Pepper Noise")

plt.xticks(x, labels)
plt.xlabel("Metode Deteksi Tepi")
plt.ylabel("Nilai Mean Squared Error (MSE)")
plt.title("Perbandingan MSE Berdasarkan Metode Deteksi Tepi")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

for bars in [bars1, bars2]:
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            y,
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()
plt.show()
