
"""
RBE/CS Spring 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code
hverma@wpi.edu

Original Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import os
import copy
import argparse

def read_images(setPath = "../Data/Train/"):
    
    files = [f for f in os.listdir(setPath) if f.endswith('.jpg')][:1]
    files = sorted(files, key=lambda filename: int(filename.split(".")[0]))
    print("Loading Images")
    images = [cv2.imread(setPath + file) for file in files]
    # grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    # grays = [cv2.resize(image, (320,240)) for image in images]
    return images

def generate_patch(image, stride, patch_size=128, p=32):
    max_x_top_left = image.shape[1] - p - stride
    max_y_top_left = image.shape[0] - p - stride
    
    new_x_top_left = np.random.randint(p, max_x_top_left - patch_size)
    new_y_top_left = np.random.randint(p, max_y_top_left - patch_size)
    
    points_A = [[new_x_top_left, new_y_top_left],
              [new_x_top_left + patch_size, new_y_top_left],
              [new_x_top_left, new_y_top_left + patch_size], 
              [new_x_top_left + patch_size, new_y_top_left + patch_size]]
    
    points_B = [[point[0] + np.random.randint(-p,p), point[1] + np.random.randint(-p,p)] for point in points_A]
    
    # # Display region
    # img = copy.deepcopy(image)
    # for center in points_A:
    #     cv2.circle(img, center, 2, (255, 0, 0), 2)
    # for center in points_B:
    #     cv2.circle(img, center, 2, (0, 0, 255), 2)
    # cv2.imshow("Original Patch", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    
    H_ab = cv2.getPerspectiveTransform(np.float32(points_A), np.float32(points_B))
    H_ba = np.linalg.inv(H_ab)
    
    warp_image = cv2.warpPerspective(image, H_ba, (2*image.shape[0], 2*image.shape[1]))
    img = copy.deepcopy(warp_image)
    for center in points_A:
        cv2.circle(img, center, 2, (255, 0, 0), 2)
    for center in points_B:
        cv2.circle(img, center, 2, (0, 0, 255), 2)
    cv2.imshow("Original Patch", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    patch_A = image[points_A[0][1]:points_A[3][1],points_A[0][0]:points_A[3][0]]
    patch_B = warp_image[points_A[0][1]:points_A[3][1],points_A[0][0]:points_A[3][0]]
    
    # cv2.imshow("patch", patch_B)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    
    H4 = np.float32([[points_B[i][0]-points_A[i][0], points_B[i][1]-points_A[i][1]] for i in range(len(points_A))])

    patch = np.dstack([patch_A, patch_B])

    return patch, H4
 
def arg_parser():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Wrapper to generate data and train.")

    # Define command-line arguments
    parser.add_argument("-generate_data", type=bool, default=True, help="Generate Data. Default: True")
    parser.add_argument("-n_patches", type=int, default=10, help="Number of patches to generate per image")
    parser.add_argument("-model", type=str, default="sup", help="[sup, unsup]")
    parser.add_argument("-data", type=str, default="Val", help="[Train, Val]")
 
    return parser

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    args = arg_parser().parse_args()

    # Access the parsed arguments
    generate_data = args.generate_data
    n_patches = args.n_patches
    data_src = args.data

    """
    Read a set of images for Panorama stitching
    """
    path = f"../Data/{data_src}/"
    if generate_data:
        path = '../Data/Val/'
        images = read_images(path)

        # to ensure each patch is not from the same region
        stride = 0
        # we want to create n patches per image
        for i in range(n_patches):
            patches_i, H4s_i = zip(*[generate_patch(image, stride) for image in images])
            if not os.path.exists(path + f"{i+1}/"):
                os.makedirs(path + f"{i+1}/")
            for j, patch in enumerate(patches_i, start=1):
                filename = path + f"{i+1}/{j}"
                np.save(filename, patch)
                
            H4s_i = np.array(H4s_i)
            np.save(path + f"{i+1}/H4s.npy", H4s_i)
            stride += 20
        print(f"Data Generated for {data_src}")
        return
    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
