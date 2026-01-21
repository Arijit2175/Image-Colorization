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

        