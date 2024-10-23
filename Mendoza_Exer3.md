#Exercise 1: Harris Corner Detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load an image
image = cv2.imread('/content/Kodie.png')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Convert the grayscale image to float32, which is required for Harris Corner Detection
gray = np.float32(gray)

# Step 4: Apply Harris Corner Detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Step 5: Result is dilated for marking the corners
dst = cv2.dilate(dst, None)

# Step 6: Threshold for an optimal value, marking the corners in the original image
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# Step 7: Display the original image with corners marked
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.show()
![image](https://github.com/user-attachments/assets/d8a5cc86-38a2-4bd4-ab24-9a2f558278e8)

```

#Exercise 2: HOG (Histogram of Oriented Gradients) Feature Extraction
```python
import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Step 1: Load an image
image = cv2.imread('/content/Arte.png')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply HOG descriptor to extract features
hog_features, hog_image = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True
)

# Step 4: Rescale HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Step 5: Display original image and HOG features (gradient orientations)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Original image
ax1.axis('off')
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')

# HOG features visualization
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap='gray')
ax2.set_title('HOG Features (Gradient Orientations)')

plt.show()
![image](https://github.com/user-attachments/assets/6bb1d88f-99a6-4604-be93-6d30ccce518b)

```
#Exercise 3: FAST (Features from Accelerated Segment Test) Keypoint Detection
```python
import cv2
import matplotlib.pyplot as plt

# Step 1: Load an image
image = cv2.imread('/content/kodie2.png')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply FAST keypoint detection
fast = cv2.FastFeatureDetector_create()

# Detect keypoints using FAST
keypoints = fast.detect(gray, None)

# Step 4: Draw the detected keypoints on the original image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Step 5: Display the image with keypoints
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('FAST Keypoint Detection')
plt.axis('off')
plt.show()

# Optionally: Display FAST settings and the number of keypoints detected
print(f"Number of keypoints detected: {len(keypoints)}")
print("FAST Settings:")
print(f"Threshold: {fast.getThreshold()}")
print(f"Non-max Suppression: {fast.getNonmaxSuppression()}")
print(f"Type of keypoints detected: {fast.getType()}")
![image](https://github.com/user-attachments/assets/8a98f43d-bc39-45c0-a7a7-b55f27cbc344)

```
#Exercise 4: Feature Matching using ORB and FLANN
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load two images
image1 = cv2.imread('/content/Arte.png')
image2 = cv2.imread('/content/Aarte.png')

# Step 2: Convert both images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Step 3: Detect keypoints and compute descriptors using ORB
orb = cv2.ORB_create()

# Detect keypoints and descriptors in both images
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Step 4: Use FLANN-based matcher for feature matching
index_params = dict(algorithm=6,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Step 5: Apply ratio test to filter good matches (Lowe's ratio test)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Step 6: Visualize the matches
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched features
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB))
plt.title('Feature Matching using ORB and FLANN')
plt.axis('off')
plt.show()

print(f"Number of good matches: {len(good_matches)}")
![image](https://github.com/user-attachments/assets/3e914a7f-94bb-469f-9885-8087d3b26687)

```
#Exercise 5: Image Segmentation using Watershed Algorithm
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load an image
image = cv2.imread('/content/kodie2.png')

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply a threshold to convert the image to binary
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: Remove noise using morphological transformations
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 5: Identify sure background and sure foreground areas
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform for sure foreground areas
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Step 6: Identify unknown regions
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 7: Label markers (connected components)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Step 8: Apply the Watershed algorithm
markers = cv2.watershed(image, markers)

# Step 9: Draw boundaries where the Watershed algorithm detected edges
image[markers == -1] = [0, 0, 255]

# Step 10: Display the results
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image Segmentation using Watershed Algorithm')
plt.axis('off')
plt.show()
![image](https://github.com/user-attachments/assets/e6415655-a422-46bb-b112-d2ead92a8ff9)
```
#Conclusion
```python
These exercises demonstrate various essential computer vision techniques, including feature detection, feature matching,
and image segmentation. In Harris Corner Detection, corners in an image are identified by highlighting areas with significant
intensity changes.HOG extracts features by analyzing gradient orientations, which is useful for object recognition. FAST keypoint
detection quickly identifies image keypoints, while ORB and FLANN efficiently match features between two images. Lastly, the Watershed
algorithm segments an image by identifying distinct regions based on intensity gradients. Together,  these methods illustrate the
versatility of image processing and feature analysis techniques for object detection, matching, and segmentation.

