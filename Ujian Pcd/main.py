import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def thresholding_correction(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold_img

def brightness_correction(img):
    brightness = 50
    corrected_img = cv2.add(img, brightness)
    return corrected_img

def show_image(img, x, y, title):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=img)
    label.image = img
    label.place(x=x, y=y)
    title_label = tk.Label(root, text=title)
    title_label.place(x=x, y=y-20)

def process_image(method):
    global original_img
    if method == 'grayscale':
        corrected_img = grayscale(original_img)
        show_image(corrected_img, 360, 170, 'Grayscale')
    elif method == 'thresholding_correction':
        corrected_img = thresholding_correction(original_img)
        show_image(corrected_img, 620, 170, 'Thresholding Correction')
    elif method == 'brightness':
        corrected_img = brightness_correction(original_img)
        show_image(corrected_img, 880, 170, 'Brightness Corretion')

def show_creator():
    creator_label = tk.Label(root, text='Nama: Ade Sinta   | Nim : F55121062   | Kelas : B')
    creator_label.place(x=480, y=100)

def open_image():
    global original_img
    file_path = filedialog.askopenfilename()
    if file_path:
        original_img = cv2.imread(file_path)
        original_img = cv2.resize(original_img, (250, 250))
        show_image(original_img, 70, 170, 'Gambar Original')
        size_label.config(format(original_img.shape[1], original_img.shape[0]))

root = tk.Tk()
root.geometry('1200x900')
root.title('UJIAN PCD')

title_label = tk.Label(root, text='Original image')
title_label.place(x=50, y=20)

open_button = tk.Button(root, text='Upload Gambar', command=open_image)
open_button.place(x=50, y=50)

correction_box = tk.LabelFrame(root, text='Metode Perbaikan Citra', padx=5, pady=5)
correction_box.place(x=50, y=550, width=1100, height=150)

grayscale_button = tk.Button(correction_box, text='grayscale', command=lambda: process_image('grayscale'))
grayscale_button.pack(side=tk.LEFT, padx=5)

thresholding_correction_button = tk.Button(correction_box, text='thresholding_correction', command=lambda: process_image('thresholding_correction'))
thresholding_correction_button.pack(side=tk.LEFT, padx=5)

brightness_button = tk.Button(correction_box, text='Brightness', command=lambda: process_image('brightness'))
brightness_button.pack(side=tk.LEFT, padx=5)

result_box = tk.LabelFrame(root, text='Output Perbaikan Citra', padx=5, pady=5)
result_box.place(x=50, y=100, width=1100, height=450)

show_creator()
root.mainloop()