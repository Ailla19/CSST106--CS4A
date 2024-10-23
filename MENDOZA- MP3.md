#Feature Extraction and Object Detection
```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the two images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Convert the images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT and ORB detectors (no SURF)
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

# Detect keypoints and descriptors using SIFT
keypoints_sift_1, descriptors_sift_1 = sift.detectAndCompute(image1_gray, None)
keypoints_sift_2, descriptors_sift_2 = sift.detectAndCompute(image2_gray, None)

# Detect keypoints and descriptors using ORB
keypoints_orb_1, descriptors_orb_1 = orb.detectAndCompute(image1_gray, None)
keypoints_orb_2, descriptors_orb_2 = orb.detectAndCompute(image2_gray, None)

# Visualize the keypoints
image1_sift = cv2.drawKeypoints(image1, keypoints_sift_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_sift = cv2.drawKeypoints(image2, keypoints_sift_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

image1_orb = cv2.drawKeypoints(image1, keypoints_orb_1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_orb = cv2.drawKeypoints(image2, keypoints_orb_2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Use cv2_imshow to display images in Colab
cv2_imshow(image1_sift)
cv2_imshow(image2_sift)
cv2_imshow(image1_orb)
cv2_imshow(image2_orb)

![image](https://github.com/user-attachments/assets/13dd6583-5bd1-4898-8225-ef854d52d12d)
![image](https://github.com/user-attachments/assets/53725acb-9768-4d3c-aa0a-190832871702)
![image](https://github.com/user-attachments/assets/c74df57f-4037-44e6-8ee3-e21fc3a4f850)
![image](https://github.com/user-attachments/assets/68f0dfcb-b1f7-450e-8ce7-0f072db3f255)

```

#Brute-Force Matcher and FLANN
```python
import cv2

# Load the two images again (ensure they are loaded in your workspace)
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Brute-Force matcher with SIFT
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# SIFT Matching
matches_sift = bf.match(descriptors_sift_1, descriptors_sift_2)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)

# Brute-Force matcher with ORB
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ORB Matching
matches_orb = bf_orb.match(descriptors_orb_1, descriptors_orb_2)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# FLANN matcher parameters for SIFT
index_params = dict(algorithm=1, trees=5)  # FLANN parameters for KDTree
search_params = dict(checks=50)  # The number of times the tree is recursively traversed

flann = cv2.FlannBasedMatcher(index_params, search_params)

# SIFT Matching using FLANN
matches_sift_flann = flann.knnMatch(descriptors_sift_1, descriptors_sift_2, k=2)

# Filter matches using Lowe's ratio test for SIFT
good_matches_sift = []
for m, n in matches_sift_flann:
    if m.distance < 0.7 * n.distance:
        good_matches_sift.append(m)

# FLANN parameters for ORB
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = {}

flann_orb = cv2.FlannBasedMatcher(index_params, search_params)

# ORB Matching using FLANN
matches_orb_flann = flann_orb.knnMatch(descriptors_orb_1, descriptors_orb_2, k=2)

# Filter matches using Lowe's ratio test for ORB
good_matches_orb = []
for m, n in matches_orb_flann:
    if m.distance < 0.7 * n.distance:
        good_matches_orb.append(m)

# Extract the matched keypoints' coordinates
if len(good_matches_sift) > 4:
    src_pts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in good_matches_sift]).reshape(-1, 1, 2)

    # Calculate the Homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the first image to align it with the second image
    height, width, channels = image2.shape
    aligned_image = cv2.warpPerspective(image1, H, (width, height))

    # Display the aligned image
    cv2_imshow(aligned_image)

    # Draw matches
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # Green matches
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Only draw inliers
                       flags=2)
    matched_img = cv2.drawMatches(image1, keypoints_sift_1, image2, keypoints_sift_2, good_matches_sift, None, **draw_params)

    cv2_imshow(matched_img)
else:
    print("Not enough good matches to compute homography.")

![image](https://github.com/user-attachments/assets/6c536570-bef7-4bb1-96fe-203ac6cb0716)
![image](https://github.com/user-attachments/assets/6273c5f7-f0c9-4060-890b-281897dea929)

```

# Function to compute homography and warp image
```python
from google.colab.patches import cv2_imshow

# Function to compute homography and warp image
def align_images(keypoints1, keypoints2, matches, img1, img2):
    # Extract matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get the size of the second image
    height, width = img2.shape[:2]

    # Warp the first image to align with the second image
    aligned_img = cv2.warpPerspective(img1, H, (width, height))

    return aligned_img, H

# Example with SIFT matches
aligned_image_sift, H_sift = align_images(keypoints_sift_1, keypoints_sift_2, matches_sift, image1, image2)

# Show the aligned image using cv2_imshow
cv2_imshow(aligned_image_sift)
![image](https://github.com/user-attachments/assets/fdf9d8b0-2766-4832-b6b0-004e8c45846d)

```
# Time SIFT alignment
```python
import time
start_time_sift = time.time()

# Perform SIFT alignment (include SIFT code here)

end_time_sift = time.time()
print(f"SIFT alignment took {end_time_sift - start_time_sift:.4f} seconds.")

# Time ORB alignment
start_time_orb = time.time()

# Perform ORB alignment (include ORB code here)

end_time_orb = time.time()
print(f"ORB alignment took {end_time_orb - start_time_orb:.4f} seconds.")

![image](https://github.com/user-attachments/assets/5faa0f6d-211f-43bc-ab4d-6e509a5b3791)
```
#Overall Understanding
```python
 SIFT and ORB. SIFT is accurate but slower, while ORB is faster and suitable for real-time tasks.
 After detecting keypoints in both images, the code matches them using two techniques. Brute-Force and FLANN.
 The matched points are then used to calculate a transformation (homography) that aligns one image with another.
