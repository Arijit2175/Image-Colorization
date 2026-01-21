import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from cv2 import dnn
import os
from threading import Thread

class ImageColorizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Colorization Tool")
        self.root.geometry("1200x700")
        
        self.proto_file = "Models/colorization_deploy_v2.prototxt"
        self.model_file = "Models/colorization_release_v2.caffemodel"
        self.pts_file = "Models/pts_in_hull.npy"

        try:
            self.net = dnn.readNetFromCaffe(self.proto_file, self.model_file)
            self.kernel = np.load(self.pts_file)
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.model_loaded = False
        
        self.current_image = None
        self.current_image_path = None
        
        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)
        
        self.browse_btn = tk.Button(top_frame, text="Browse Image", command=self.browse_image, 
                                     width=15, height=2, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.colorize_btn = tk.Button(top_frame, text="Colorize", command=self.colorize_image, 
                                       width=15, height=2, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.colorize_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(top_frame, text="Save Result", command=self.save_image, 
                                   width=15, height=2, bg="#FF9800", fg="white", font=("Arial", 10, "bold"))
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="Ready to load image", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.original_frame = tk.LabelFrame(main_frame, text="Original Image", font=("Arial", 11, "bold"), padx=5, pady=5)
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_label = tk.Label(self.original_frame, bg="gray", width=400, height=350)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        
        self.colorized_frame = tk.LabelFrame(main_frame, text="Colorized Image", font=("Arial", 11, "bold"), padx=5, pady=5)
        self.colorized_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.colorized_label = tk.Label(self.colorized_frame, bg="gray", width=400, height=350)
        self.colorized_label.pack(fill=tk.BOTH, expand=True)
        
        self.colorize_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            self.display_image(self.current_image, self.original_label)
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.colorize_btn.config(state=tk.NORMAL)

    def display_image(self, cv_image, label):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        h, w = rgb_image.shape[:2]
        max_width, max_height = 380, 320
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        label.config(image=photo)
        label.image = photo

    def colorize_image(self):
        if self.current_image is None or not self.model_loaded:
            messagebox.showerror("Error", "Please load an image first")
            return
        
        self.status_label.config(text="Colorizing... Please wait")
        self.colorize_btn.config(state=tk.DISABLED)
        self.root.update()
        
        thread = Thread(target=self._colorize_worker)
        thread.start()

    def _colorize_worker(self):
        try:
            scaled = self.current_image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            
            class8 = self.net.getLayerId("class8_ab")
            conv8 = self.net.getLayerId("conv8_313_rh")
            pts = self.kernel.transpose().reshape(2, 313, 1, 1)
            self.net.getLayer(class8).blobs = [pts.astype("float32")]
            self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50
            
            self.net.setInput(dnn.blobFromImage(L))
            ab_channel = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab_channel = cv2.resize(ab_channel, (self.current_image.shape[1], self.current_image.shape[0]))
            
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
            
            self.colorized_image = cv2.cvtColor(colorized.astype("uint8"), cv2.COLOR_LAB2BGR)
  
            self.root.after(0, self._display_colorized_result)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Colorization failed: {e}"))
        finally:
            self.root.after(0, lambda: self.colorize_btn.config(state=tk.NORMAL))

    

