# 🎨 Image Colorization Tool using OpenCV & Deep Learning

Transform old black-and-white photos into vibrant, realistic color images using **Deep Learning + OpenCV**, wrapped in a modern **Tkinter GUI** with advanced post-processing enhancements.
This project leverages a pretrained CNN-based colorization model and applies multiple post-processing techniques to significantly improve visual quality.

---

## ✨ Features

- 🖼️ **Black & White Image Colorization**
- 🧠 **Deep Learning-based Color Prediction (OpenCV DNN)**
- 🎛️ **Advanced Post-Processing Pipeline**
- 🧴 Noise reduction & edge-preserving smoothing
- 🌈 Contrast enhancement using CLAHE
- 🎨 Controlled saturation boosting
- 🧑 Skin tone detection & correction
- 🔍 Advanced sharpening for fine details
- 📈 Super-resolution upscaling
- 🧵 Multi-threaded processing (UI never freezes)
- 💾 Save colorized images in high quality
- 🎨 Modern dark-themed GUI built with Tkinter

---

## 🧠 Methodology

1. Input image is converted from **BGR → LAB color space**
2. **L channel (grayscale)** is fed to a pretrained CNN
3. Model predicts **A & B color channels**
4. LAB image is reconstructed and converted back to BGR
5. Post-processing improves realism and perceptual quality

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV (DNN module)**
- **NumPy**
- **Tkinter**
- **Pillow (PIL)**

---

## 📂 Project Structure

```
Image-Colorization/
│
├── Models/
│ ├── colorization_deploy_v2.prototxt
│ ├── colorization_release_v2.caffemodel
│ └── pts_in_hull.npy
│
├── colorization_gui.py
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1️⃣ Clone the repository
```
git clone "url"
cd image-colorization-tool
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Download the pretrained model
Place the following files inside the **Models/** directory:

- colorization_deploy_v2.prototxt

- colorization_release_v2.caffemodel

- pts_in_hull.npy

* ℹ️ See the References section below for official download links.

---

## ▶️ Usage

Run the application:

```
python colorization_gui.py
```

### Workflow

1) 📁 Click Browse Image
2) ✨ Click Colorize
3) 🎉 Preview the colorized output
4) 💾 Save the result

---

## 🎯 Post-Processing Pipeline

To improve realism, the following enhancements are applied after colorization:
1) 🔇 Noise reduction (Non-local Means)
2) 🌗 Contrast enhancement (CLAHE on L channel)
3) 🌈 Saturation boost (HSV space)
4) 🧠 Edge-preserving smoothing (Bilateral Filter)
5) 🧑 Skin tone detection & correction
6) 🔍 Advanced sharpening
7) 📈 Super-resolution upscaling

This significantly reduces:
- Washed-out colors
- Color bleeding
- Flat contrast
- Unrealistic skin tones

---

## 🖼️ Preview (Results)

### Original vs Colorized Output

| Original Black & White Image | Colorized Image |
|------------------------------|-----------------|
| ![Original](assets/original.jpg) | ![Colorized](assets/colorized.jpg) |

---

