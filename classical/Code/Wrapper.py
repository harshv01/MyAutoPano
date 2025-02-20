
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 1: Classical CV
hverma@wpi.edu

Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute

Reference for cylindrical transformation
https://www.scribd.com/document/510892500/Panorama-Stitching
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here

import argparse
import glob

from skimage.feature import peak_local_max

   
## --------- Functions -----------

def Cimg_gen(img, blocksize, ksize, k):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = np.float32(bw_img)

    dst = cv2.cornerHarris(bw_img, blocksize, ksize, k)
    return dst

def ANMS(corner_score_image, num_best_features, mask=None):

    if mask is not None:
        corner_score_image = corner_score_image*mask
        
    coordinates = peak_local_max(corner_score_image, min_distance=10)

    best_list = []

    for i in range(len(coordinates)):
        y_i, x_i = coordinates[i]
        # print('i = ', i)
        r_i = float('inf')

        for j in range(len(coordinates)):
            y_j, x_j = coordinates[j]
            if corner_score_image[y_i][x_i] > corner_score_image[y_j][x_j]:
                ED = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
                if ED < r_i:
                    r_i = ED

        best_list.append((r_i, x_i, y_i))

    best_list.sort(key=lambda x: x[0], reverse=True)
    if len(best_list) < num_best_features:
        num_best_features = len(best_list)


    Nbest_list = [(tup[1], tup[2]) for tup in best_list]
    

    # color_image_with_circles = color_image.copy()

    # color_image = cv2.cvtColor(corner_score_image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # for x, y in Nbest_list[:num_best_features]:
    #     cv2.circle(color_image, (x, y), radius=5, color=(0, 0, 255), thickness=2)

    # cv2.imshow('ANMS', color_image)

    return Nbest_list[:num_best_features]       # [(pixel_x, pixel_y), (), ().....]
    
def feature_descriptor(img, Nbest_list):
    feature_vectors = []
    dic ={}
    for i in range(len(Nbest_list)):
        
        # to handle corner case of a point not having 20 pixels in the left or upper directions
        padded_image = np.pad(img, ((20,20), (20,20)))


        patch = np.copy(padded_image[ (Nbest_list[i][1]+20) -20 : (Nbest_list[i][1]+20) +20 , Nbest_list[i][0]+20 -20 : Nbest_list[i][0]+20 +20 ])
        
        # print(patch.shape)
        
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        # print(patch.shape)

        subsampled_patch = patch[0::5, 0::5]       # Take every 5th row and column to build 8x8 subsample
        # print(subsampled_patch.shape)

        # subsampled_patch = cv2.resize(patch, (8,8), interpolation = cv2.INTER_AREA)

        patch_vector = subsampled_patch.flatten()

        standardized = (patch_vector-np.mean(patch_vector))/np.std(patch_vector)
        # feature_vectors.append((standardized,Nbest_list[i]))
        # print(np.mean(standardized))
        feature_vectors.append((standardized, Nbest_list[i]))
        
   
    temp = np.array(feature_vectors, dtype=object)

    return feature_vectors  # Output format [ (feature array, (pixel_x, pixel_y)), (), (),.....]

def compute_homography(point_pairs):
    # Create matrix A
    points1, points2 = zip(*point_pairs)
    A = []
    for i in range(len(points1)):
        x,y = points1[i]
        x_dash, y_dash = points2[i]
        # x*h11 + y*h12 + h13 + 0*(h21 + h22 + h23) - x'*x*h31 - x'*y*h32 - x'*h33 = 0
        A.append([-x, -y, -1, 0, 0, 0, x_dash*x , x_dash*y, x_dash])
        # 0*(h11 + h12 + h13) + x*h21 + y*h22 + h23 - y'*x*h31 - y'*y*h32 - y'*h33 = 0
        A.append([0, 0, 0, -x, -y, -1, y_dash*x, y_dash*y, y_dash])

    A = np.array(A)

    # least squares method. Ax =0
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    return H

def RANSAC(feature_pairs, threshold, max_iter):
    inliers = []
    iter = 0
    max_len = 0
    longest_set = None

    print(len(feature_pairs))
    while iter < max_iter and len(inliers) / len(feature_pairs) <= 0.9:

        rand_indices = np.random.randint(low=0, high=len(feature_pairs), size=4)

        p1, p1_dash = feature_pairs[rand_indices[0]]
        p2, p2_dash = feature_pairs[rand_indices[1]]
        p3, p3_dash = feature_pairs[rand_indices[2]]
        p4, p4_dash = feature_pairs[rand_indices[3]]

        pair_list = [(p1, p1_dash), (p2, p2_dash), (p3, p3_dash), (p4, p4_dash)]
        homography = compute_homography(pair_list)

        inliers = []
        for i in range(len(feature_pairs)):

            p_i, p_i_dash = feature_pairs[i]
            new_p_i = p_i+(1,)  #Add 1 to end, to cross multiply with homographic matrix

            new_p_i = np.array(new_p_i)
            H_p_i = np.dot(homography, new_p_i.T)
            
            H_p_i = H_p_i/H_p_i[2]

            H_p_i = H_p_i[:-1]  # Remove 1 from the end

            SSD = np.sum(np.square(p_i_dash - H_p_i))

            if SSD < threshold:
                inliers.append(feature_pairs[i])

        if len(inliers) > max_len:
            longest_set = inliers.copy()
            max_len = len(inliers)

        iter += 1

    print('final inlier:', longest_set)
    print('largest length',max_len)
    print('double check: ', len(longest_set))
    homography_hat = compute_homography(longest_set)
    return homography_hat, longest_set

def feature_matching(encode_vecs_img1, encode_vecs_img2, ratio):
    final_match_pairs = []
    matched_img2_points = set()

    for point1 in encode_vecs_img1:     # nested for loops to filter out many-to-one matches
        matches = []

        for point2 in encode_vecs_img2:
            if point2[1] not in matched_img2_points:
                SSD = np.sum(np.square(point1[0] - point2[0]))
                matches.append((SSD, point1, point2))

        matches.sort(key=lambda x: x[0])

        if matches:
            best_match = matches[0]

            if (best_match[0] / matches[1][0]) < ratio:
                final_match_pairs.append((best_match[1][1], best_match[2][1]))
                matched_img2_points.add(best_match[2][1])

    return final_match_pairs

def conv_coors_key(pixel_coors, size):    #convert coordinates to keypoints
    keypoints = []
    for pixel in pixel_coors:
        keypoints.append(cv2.KeyPoint(float(pixel[0]), float(pixel[1]), size))
    return keypoints

def stitcher(img1, img2, H_input):

    image1, image2 = img1.copy(), img2.copy()

    h0 ,w0 = image1.shape[0], image1.shape[1]
    h1 ,w1 = image2.shape[0], image2.shape[1]

    # get corners
    image1_corners = np.float32(np.reshape([[0, 0], [0, h0], [w0, h0], [w0, 0]], [-1,1,2]))
    img0_corners_warped = cv2.perspectiveTransform(image1_corners, H_input)    
    image2_corners = np.reshape([[0, 0], [0, h1], [w1, h1], [w1, 0]], [-1,1,2]).astype(np.float32)
    all_corners = np.concatenate((img0_corners_warped, image2_corners), axis = 0)

    # calculate min max coordinates
    temp = np.array([corners.ravel() for corners in all_corners])
    min_x, min_y = np.int0(np.min(temp, axis = 0))
    max_x, max_y = np.int0(np.max(temp, axis = 0))

    #linear transformation matrix to account for the lateral shift
    H_linear = np.array([[1,0,-min_x], [0,1,-min_y], [0,0,1]])
    
    # H1_3 = H1_2 * H2_3
    H_eval = np.dot(H_linear, H_input)

    warped_image1 = cv2.warpPerspective(image1, H_eval, (max_x-min_x, max_y-min_y))

    stitched_image = warped_image1.copy()
    stitched_image[-min_y:-min_y+h1, -min_x: -min_x+w1] = image2

    indices = np.where(image2 == [0,0,0])
    y = indices[0] + -min_y 
    x = indices[1] + -min_x 

    stitched_image[y,x] = warped_image1[y,x]
    
    return stitched_image

def xy_to_XY(x, y, center, focal_length):

    xc, yc = center[0], center[1]

    x_new = ((np.tan((x-xc)/focal_length))*focal_length)+xc
    y_new = ((y-yc)/np.cos((x-xc)/focal_length))+yc
    
    return x_new, y_new



def cylinder_transform(image):
    
    # function inspired by: https://www.scribd.com/document/510892500/Panorama-Stitching
    
    height, width = image.shape[:2]    
    f = 1200                 # Focal length
    center = [width//2, height//2]
    
    image_canvas = np.zeros(image.shape, dtype=np.uint8)
    
    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(width) for j in range(height)])

    # In order to map out all the possible coordinates, we build an array

    all_coords = []
    for i in range(width):
        for j in range(height):
            all_coords.append([i, j])

    all_coords = np.array(all_coords)

    all_x = all_coords[:, 0]
    all_y = all_coords[:, 1]
    
    # Finding corresponding coordinates of the transformed image in the initial image
    mapped_x, mapped_y = xy_to_XY(all_x, all_y, center, f)

    # Rounding off the coordinate values to get exact pixel values (top-left corner)
    rounded_mapped_x = mapped_x.astype(int)
    rounded_mapped_y = mapped_y.astype(int)

    # Finding transformed image points whose corresponding 
    # initial image points lies inside the initial image
    GoodIndices = (rounded_mapped_x >= 0) * (rounded_mapped_x <= (width-2)) * \
                  (rounded_mapped_y >= 0) * (rounded_mapped_y <= (height-2))

    # Removing all the outside points from everywhere
    all_x = all_x[GoodIndices]
    all_y = all_y[GoodIndices]
    
    mapped_x = mapped_x[GoodIndices]
    mapped_y = mapped_y[GoodIndices]

    rounded_mapped_x = rounded_mapped_x[GoodIndices]
    rounded_mapped_y = rounded_mapped_y[GoodIndices]

    # Bilinear interpolation
    xdiff = mapped_x - rounded_mapped_x
    ydiff = mapped_y - rounded_mapped_y

    weight_tl = (1.0-xdiff)*(1.0-ydiff)
    weight_tr = (xdiff)*(1.0 - ydiff)
    weight_bl = (1.0-xdiff)*(ydiff)
    weight_br = (xdiff)*(ydiff)
    
    image_canvas[all_y, all_x, :] = ( weight_tl[:, None]*image[rounded_mapped_y, rounded_mapped_x,:] ) + \
                                      ( weight_tr[:, None]*image[rounded_mapped_y, rounded_mapped_x + 1,:] ) + \
                                      ( weight_bl[:, None]*image[rounded_mapped_y+1, rounded_mapped_x, :] ) + \
                                      ( weight_br[:, None]*image[rounded_mapped_y+1, rounded_mapped_x+1, :] )


    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(all_x)

    # Cropping out the black region from both sides (using symmetricity)
    image_canvas = image_canvas[:, min_x : -min_x, :]

    return image_canvas, all_x-min_x, all_y

def main():

    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    # path = '../Data/P1TestSet/P1TestSet/Phase1/TestSet3'

    path = '../Data/Train/Set1'

    files = glob.glob(path + '/*.jpg')
    files.sort()

    print(len(files))
    
    warped_copy_left, _, _ = cylinder_transform(cv2.imread(files[0]))

    for i in range(1, len(files)):

        img1 = warped_copy_left
        img2, _, _ = cylinder_transform(cv2.imread(files[i]))

        
        """
        Corner Detection
        Save Corner detection output as corners.png
        """
        print("Corner Detection Started\n")
        bw_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        bw_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        match_pairs = []

        while len(match_pairs) == 0:       # Keep running until match pairs are not found

            """
            Perform ANMS: Adaptive Non-Maximal Suppression
            Save ANMS output as anms.png
            """ 
            
            Cimg1 = Cimg_gen(img1, 5, 3, 0.001)
            Cimg2 = Cimg_gen(img2, 5, 3, 0.001)


            Nbest_list1 = ANMS(Cimg1, num_best_features=10000)
            Nbest_list2 = ANMS(Cimg2, num_best_features=10000)

        

            """
            Feature Descriptors
            Save Feature Descriptor output as FD.png
            """
            encode_vecs_img1 = feature_descriptor(bw_img1, Nbest_list1)
            encode_vecs_img2 = feature_descriptor(bw_img2, Nbest_list2)
            
            """
            Feature Matching
            Save Feature Matching output as matching.png
            """
            match_pairs = feature_matching(encode_vecs_img1=encode_vecs_img1, encode_vecs_img2=encode_vecs_img2, ratio=0.9)

        
        print("ANMS Done\n")
        print("Feature Descriptors Done\n")
        print("Feature Matching\n")


        # Printing corners---------------------------------
        corners1 = np.column_stack(np.where(Cimg1 > 0.001 * Cimg1.max()))
        corners2 = np.column_stack(np.where(Cimg2 > 0.001 * Cimg2.max()))

        output_img1 = img1.copy()
        output_img2 = img2.copy()

        for x, y in corners1:
            cv2.circle(output_img1, (y, x), radius=1, color=(0, 0, 255), thickness=1)

        for x, y in corners2:
            cv2.circle(output_img2, (y, x), radius=1, color=(0, 0, 255), thickness=1)

        # cv2.imwrite()

        cv2.imshow('Corners 1', output_img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('Corners 2', output_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # End of printing corners-------------

        # Printing ANMS------------------------

        ANMS_1 = img1.copy()
        ANMS_2 = img2.copy()

        for x, y in Nbest_list1:
            cv2.circle(ANMS_1, (y, x), radius=1, color=(0, 0, 255), thickness=1)

        for x, y in Nbest_list2:
            cv2.circle(ANMS_2, (y, x), radius=1, color=(0, 0, 255), thickness=1)

        cv2.imshow('ANMS Points 1', ANMS_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('ANMS Points 2', ANMS_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # End of printing ANMS--------------

        """
        Refine: RANSAC, Estimate Homography
        """
        print("RANSAC Done\n")

        cv2.imshow('ANMS_1', ANMS_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow('ANMS_2', ANMS_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        pixel_coors1 = [i[0] for i in match_pairs]
        pixel_coors2 = [i[1] for i in match_pairs]
        
        keypoints_1 = conv_coors_key(pixel_coors1, 5)
        keypoints_2 = conv_coors_key(pixel_coors2, 5)

        print("Calculated Keypoints\n")

        cv2_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(match_pairs))]

        result_img = cv2.drawMatches(img1, 
                             keypoints_1, 
                             img2, 
                             keypoints_2, 
                             cv2_matches, 
                             outImg = None, 
                             matchesThickness = 1,
                             matchColor=(0, 255, 255), 
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                             )
        
        cv2.imshow('Unrefined Matches', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        H_cap, longest_set_inliers = RANSAC(match_pairs, threshold = 175, max_iter=10000)

        new_pixel_coors1 = [i[0] for i in longest_set_inliers]
        new_pixel_coors2 = [i[1] for i in longest_set_inliers]
        
        new_keypoints_1 = conv_coors_key(new_pixel_coors1, 5)
        new_keypoints_2 = conv_coors_key(new_pixel_coors2, 5)

        new_cv2_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(longest_set_inliers))]

        new_result_img = cv2.drawMatches(img1, 
                                new_keypoints_1, 
                                img2, 
                                new_keypoints_2, 
                                new_cv2_matches, 
                                outImg = None, 
                                matchesThickness = 1,
                                matchColor=(0, 255, 255), 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                )
        cv2.imshow('Refined Matches After RANSAC', new_result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        warped_copy_left = stitcher(img1, img2, H_cap)
        
    cv2.imshow('Final Merged', warped_copy_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
	main()


 


