from flask import Flask, request, jsonify
import cv2
import os
from YOLO import computeCobb  # Assuming computeCobb is defined in YOLO.py

app = Flask(__name__)

@app.route('/compute-cobb', methods=['POST'])
def compute_cobb_api():
    try:
        file = request.files['image']
        if file:
            # Save the uploaded image temporarily
            image_path = 'temp_image.jpg'
            file.save(image_path)

            # Read the uploaded image using OpenCV
            image = cv2.imread(image_path)
            cobb_up, cobb_low, img_cobb, result = computeCobb(image)

            # Remove the temporary image file
            os.remove(image_path)

            if (cobb_up or cobb_low) is None:
                return jsonify({'error': 'No vertebrae detected or wrong image'})

            if abs(cobb_up) > abs(cobb_low):
                cobb_angle = abs(cobb_up)
            else:
                cobb_angle = abs(cobb_low)

            # Return the results as JSON
            return jsonify({'cobb_angle': cobb_angle, 'classification': result})
        else:
            return jsonify({'error': 'No image uploaded'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
