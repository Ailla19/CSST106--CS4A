#Machine Problem: Object Detection and Recognition using YOLO.
```python
pip install opencv-python opencv-python-headless numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the YOLO model
def load_yolo_model(weights_path, config_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    return net

weights_path = '/content/yolov3.weights'
config_path = '/content/yolov3.cfg'
net = load_yolo_model(weights_path, config_path)

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Step 2: Load an image that contains multiple objects
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

image_path = '/content/Dos.jpg'
image = load_image(image_path)

# Step 3: Object Detection
def detect_objects(image, net):
    height, width = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    return detections, width, height


# Step 4: Visualization
def draw_boxes(image, detections, width, height):
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:] 
            class_id = np.argmax(scores) 
            confidence = scores[class_id] 

            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        indexes = indexes.flatten()

        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidences[i]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        print("No valid boxes found for this image.")

    return image


# Step 5: Test on multiple images
image_paths = [
    '/content/Dos.jpg',
    '/content/Arte.jpg',
    '/content/Kodie.jpg',
    '/content/koods.jpg',
    '/content/hachi.jpg',
    '/content/Sipi.jpg',
    '/content/Aren.jpg'
]

for img_path in image_paths:
    img = load_image(img_path)
    detections, width, height = detect_objects(img, net)
    img_with_boxes = draw_boxes(img.copy(), detections, width, height)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.title(f'Object Detection on {img_path}')
    plt.show()



![image](https://github.com/user-attachments/assets/912ebcf9-e363-4a10-b183-cbff3eb89482)
![image](https://github.com/user-attachments/assets/6f56b92f-f33e-4c4d-b659-400701e0a93d)
![image](https://github.com/user-attachments/assets/98ec724c-a3e0-4a5e-9c91-c1832ac083de)
![image](https://github.com/user-attachments/assets/00765c33-30fb-431d-9d11-df30dbe460d1)




```
