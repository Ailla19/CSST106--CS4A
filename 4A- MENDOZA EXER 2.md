# SIFT FEATURE EXTRACTION
```python
import cv2
import matplotlib.pyplot as plt

#Load the image

image = cv2.imread('/content/image1.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initialize SIFT detector

sift = cv2.SIFT_create()

#Detect keypoints and descriptors

keypoints, descriptors = sift.detectAndCompute(gray_image, None)

#Draw keypoints on the image

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

#Display the image with keypoints

plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.show()

![image](https://github.com/user-attachments/assets/8d814e43-9555-4b76-a069-e8e3b0c83e73)

```


#ORB Feature Extraction
```python
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/image2.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints')
plt.show()
![image](https://github.com/user-attachments/assets/d903d4ce-7bc4-4ae0-bbf8-b145e696543d)

```

#Feature Matching using SIFT
```python
# Load two images
image1 = cv2.imread('/content/image1.png')
image2 = cv2.imread('/content/image2.png')

#Initialize SIFT detector
sift = cv2.SIFT_create()

#Find keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

#Initialize the matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#Match descriptors
matches = bf.match(descriptors1, descriptors2)

#Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

#Draw matches
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Display the matches
plt.imshow(image_matches)
plt.title('Feature Matching with SIFT')
plt.show()
![image](https://github.com/user-attachments/assets/a6e80138-a0e2-4cd5-8446-f55b32109f84)
```


#Real-World Applications (Image Stitching using Homography)
```python
import cv2
import numpy as np  # Import numpy
import matplotlib.pyplot as plt

# Load two images
image1 = cv2.imread('/content/image1.png')
image2 = cv2.imread('/content/image2.png')

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract location of good matches
src_pts = np.float32(
    [keypoints1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)
dst_pts = np.float32(
    [keypoints2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# Find homography matrix
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp one image to align with the other
h, w, _ = image1.shape
result = cv2.warpPerspective(image1, M, (w, h))

# Display the result
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Image Alignment using Homography')
plt.show()
![image](https://github.com/user-attachments/assets/3b5862d2-c35c-412e-88c6-531277c91ed2)
```


#Combining SIFT and ORB
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
image1 = cv2.imread('/content/image1.png')
image2 = cv2.imread('/content/image2.png')

# Convert to grayscale (SIFT and ORB require grayscale images)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# SIFT detector
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(gray1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(gray2, None)

# ORB detector
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(gray1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(gray2, None)

# BFMatcher for SIFT (uses L2 norm)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(descriptors1_sift, descriptors2_sift)

# BFMatcher for ORB (uses Hamming distance)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)

# Sort matches by distance (for better matching results)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# Draw top matches for SIFT
sift_result = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_sift[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Draw top matches for ORB
orb_result = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_orb[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot SIFT matching results
plt.figure(figsize=(12, 6))
plt.title("SIFT Keypoints Matching")
plt.imshow(cv2.cvtColor(sift_result, cv2.COLOR_BGR2RGB))
plt.show()

# Plot ORB matching results
plt.figure(figsize=(12, 6))
plt.title("ORB Keypoints Matching")
plt.imshow(cv2.cvtColor(orb_result, cv2.COLOR_BGR2RGB))
plt.show()
![image](https://github.com/user-attachments/assets/1f736d3e-c74b-4f95-9610-237fbfd02258)
![image](https://github.com/user-attachments/assets/746edfff-3227-44bf-967d-f57091c6dda8)



```
#Overall Understanding
SIFT is known for accurately detecting and matching keypoints, even in images that have been scaled
or rotated. SURF is faster than SIFT but still good at finding keypoints. ORB is highly efficient and
ideal for real-time applications. Feature matching helps compare different images to identify
common objects or align them. Homography is used to align images, like stitching them together to create a panorama.

