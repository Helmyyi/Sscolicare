from flask import Flask, request, jsonify
import cv2
import os
from YOLO import computeCobb  # Assuming computeCobb is defined in YOLO.py
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/compute-cobb', methods=['POST'])
def compute_cobb_api():
    image_path = 'temp_image.jpg'
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        # Save the uploaded image temporarily
        file.save(image_path)

        # Read the uploaded image using OpenCV
        image = cv2.imread(image_path)

        # Ensure the image is read correctly
        if image is None:
            logger.error('Invalid image file')
            return jsonify({'error': 'Invalid image file'}), 400

        # Perform Cobb angle computation
        cobb_up, cobb_low, img_cobb, result = computeCobb(image)

        # Check the results of the computation
        if cobb_up is None or cobb_low is None:
            logger.error('No vertebrae detected or wrong image')
            return jsonify({'error': 'No vertebrae detected or wrong image'}), 400

        cobb_angle = max(abs(cobb_up), abs(cobb_low))

        # Return the results as JSON
        return jsonify({'cobb_angle': cobb_angle, 'classification': result})

    except Exception as e:
        logger.exception('Error processing image')
        return jsonify({'error': str(e)}), 500

    finally:
        # Ensure the temporary image file is removed
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port, debug=True)
