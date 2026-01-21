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

