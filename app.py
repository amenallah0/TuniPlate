import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_yolo(weights_path, config_path):
    """Load YOLO model from weights and configuration files."""
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names_all = net.getLayerNames()
    layer_names_output = [layer_names_all[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, layer_names_output

def detect_license_plate(image_path, net, layer_names_output, labels, probability_minimum=0.5, threshold=0.3):
    """Detect license plates in the image using YOLO."""
    image = cv2.imread(image_path)
    if image is None:
        return None, []

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names_output)

    bounding_boxes, confidences, class_numbers = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum and labels[class_current] == 'license_plate':
                box = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    detected_plates = []

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            plate_image = image[y_min:y_min + box_height, x_min:x_min + box_width]
            detected_plates.append((plate_image, (x_min, y_min, box_width, box_height)))

    return image, detected_plates

@app.route('/api/process-image', methods=['POST'])
def process_image():
    data = request.get_json()
    file_path = data['file_path']

    weights_path = 'yolo/lapi.weights'
    config_path = 'yolo/darknet-yolov3.cfg'
    labels = ['license_plate']

    net, layer_names_output = load_yolo(weights_path, config_path)
    image, detected_plates = detect_license_plate(file_path, net, layer_names_output, labels)

    if not detected_plates:
        return jsonify({"message": "No license plate detected."}), 404

    plate_img, (x_min, y_min, box_width, box_height) = detected_plates[0]

    # Apply blur to the license plate region
    blurred_image = image.copy()
    blurred_image[y_min:y_min + box_height, x_min:x_min + box_width] = cv2.GaussianBlur(plate_img, (99, 99), 0)
    cv2.imwrite(file_path, blurred_image)

    return jsonify({"message": "License plate blurred successfully.", "file_path": file_path}), 200

if __name__ == '__main__':
    app.run(debug=True)
