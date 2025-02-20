# MyAutoPano
Installation Guide for MyAutoPano: Phase 2

## Overview

The purpose of this project is to stitch two or more images in order to create one seamless panorama image. This project is a part of the course RBE549 Computer Vision at Worcester Polytechnic Institute, Spring 2024 semester. 

**Stitches upto 8 images using cylindrical projection technique!**
<br>
<br>
<img src="https://github.com/user-attachments/assets/d28fb46c-3540-43ac-be15-8eaa5edaef9e" width="500">

**Feature matching:**
<br>
<br>
<img src="https://github.com/user-attachments/assets/d1b1c086-84dc-4a65-b401-414b5b0b3a5c" height="200">
<br>
<img src="https://github.com/user-attachments/assets/0d434f6d-b70f-46b3-bdef-628cb1118a7c" height="200">
<br>
<img src="https://github.com/user-attachments/assets/2cb0a347-5037-4140-aeb3-7b37484c6897" height="200">


## Required Libraries

To successfully run the script, you need to install the following Python libraries:

NumPy - Provides support for large multidimensional arrays and matrices, along with mathematical functions.

```pip install numpy```

OpenCV (cv2) - Used for image processing, feature extraction, and transformation.

```pip install opencv-python```

scikit-image (skimage) - Used for image processing tasks such as feature detection.

```pip install scikit-image```

matplotlib - Required for visualizing images and plotting results.

```pip install matplotlib```

## Installation Instructions

To install all dependencies at once, run:

```pip install numpy opencv-python scikit-image matplotlib```

## Running the Script

To execute the scripts in 'classical' or 'deep_learning', run the following command:

```python Wrapper.py```
