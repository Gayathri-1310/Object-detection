from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, World!'

def process_image(image_path):
    thres = 0.45  # Threshold to detect object
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

    classNames = []
    with open('../flask_py/coco.names', 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = r'../flask_py/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = r'../flask_py/frozen_inference_graph.pb'

    img = cv2.imread(image_path)
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    vehicle_counts = {vehicle: 0 for vehicle in vehicle_classes}
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in vehicle_classes:
                vehicle_counts[className] += 1

    # Draw bounding boxes on the image
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    # Save the detection image
    detection_image_path = 'result/detection_' + os.path.basename(image_path)
    cv2.imwrite(detection_image_path, img)

    # Prepare data for response
    car_count = vehicle_counts['car']
    truck_count = vehicle_counts['truck']
    bus_count = vehicle_counts['bus']
    motorcycle_count = vehicle_counts['motorcycle']
    bicycle_count = vehicle_counts['bicycle']
    response_data = {
        'car_count': car_count,
        'truck_count': truck_count,
        'bus_count': bus_count,
        'motorcycle_count': motorcycle_count,
        'bicycle_count': bicycle_count,
        'original_image_path': '/result/' + os.path.basename(image_path),
        'detection_image_path': '/result/' + os.path.basename(detection_image_path)
    }

    return jsonify(response_data)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/fileUpload', methods=['POST'])
def file_upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        try:
            image_path = 'uploads/' + file.filename
            file.save(image_path)
            # processed_image = process_image(image_path)

            return process_image(image_path)
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return 'File type not allowed'

@app.route('/result/<path:filename>')
def get_result(filename):
    return send_from_directory('result', filename)

if __name__ == '__main__':
    app.run(debug=True)
