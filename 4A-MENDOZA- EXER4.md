#Object Detection and Recognition

#Exercise 1: HOG (Histogram of Oriented Gradients) Object Detection
```python
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('/content/Arte.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply HOG descriptor
features, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True)

# Display the HOG image
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(hog_image, cmap='gray')
plt.show()
![image](https://github.com/user-attachments/assets/9d4ba766-9ea1-449d-9b9b-241efcb1ad19)

```
#Exercise 2: YOLO (You Only Look Once) Object Detection
```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load YOLO model and configuration
net = cv2.dnn.readNet('/content/yolov3.weights', '/content/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open('/content/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load an image
image = cv2.imread('/content/piiii.jpg')

if image is None:
    print("Error: Image not found or unable to load.")
else:
    height, width, channels = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (10, 155, 10), 2)

                # Put label with confidence
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 155, 10), 2)

    # Display the image using cv2_imshow
    cv2_imshow(image)
![image](https://github.com/user-attachments/assets/e3eb15ee-dedc-4355-818b-483a9de01e2e)

```
#Exercise 3: SSD (Single Shot MultiBox Detector) with TensorFlow
```python
!wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
!tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

![image](https://github.com/user-attachments/assets/2537086a-42c3-4ed6-b119-f09699e050b2)

import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow

# Load pre-trained SSD model
model = tf.saved_model.load('/content/ssd_mobilenet_v2_coco_2018_03_29/saved_model')

# Load the 'serving_default' signature
detection_fn = model.signatures['serving_default']

# Load image
image_path = '/content/Arte.jpg'
image_np = cv2.imread(image_path)

# Convert the image to a tensor and add a batch dimension
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

# Run the detection function
detections = detection_fn(input_tensor)

# Extract detection details
detection_boxes = detections['detection_boxes'][0].numpy()
detection_scores = detections['detection_scores'][0].numpy()

# Visualize the bounding boxes
for i in range(detection_boxes.shape[0]):
    if detection_scores[i] > 0.5:
        ymin, xmin, ymax, xmax = detection_boxes[i]
        (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                      ymin * image_np.shape[0], ymax * image_np.shape[0])

        # Draw bounding box on the image
        cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)

# Display the image with bounding boxes
cv2_imshow(image_np)
![image](https://github.com/user-attachments/assets/1e5d21ad-aff5-4879-85e1-bd20a39a60b1)


```
#Exercise 4: Traditional vs. Deep Learning Object Detection Comparison
```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load pre-trained SVM model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load image
image = cv2.imread('/content/Arte.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform HOG-SVM based object detection
(rects, weights) = hog.detectMultiScale(gray_image, winStride=(8, 8), padding=(16, 16), scale=1.05)

# Draw bounding boxes for detected objects
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output using cv2_imshow
cv2_imshow(image)


# Load YOLO model and configuration
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load an image
image = cv2.imread('/content/Arte.jpg')
height, width, channels = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image using cv2_imshow
cv2_imshow(image)
![image](https://github.com/user-attachments/assets/dc14418d-90dc-4e82-a33d-a301d405cb5e)
![image](https://github.com/user-attachments/assets/8bc383ea-9434-4b44-8853-a76a8abe83fa)

```

#Conclusion
```python
In comparing traditional object detection methods like HOG-SVM with deep learning-based approaches like YOLO and SSD,
there are clear differences in both accuracy and speed. Traditional methods, such as HOG-SVM,are more rigid and tend
to work best for detecting specific objects like pedestrians, but their accuracy may diminish in complex or cluttered
scenes. On the other hand, deep learning methods like YOLO and SSD are more robust, capable of detecting multiple object
classes simultaneously with higher accuracy, and are more adaptable to varying lighting conditions and object sizes.
YOLO, in particular, excels in real-time applications due to its speed, while SSD offers a good balance between speed and
accuracy. Overall, deep learning methods outperform traditional approaches in object detection tasks, especially for complex,
real-world environments.
```
