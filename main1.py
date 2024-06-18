import os
import re
import cv2
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
from ultralytics import YOLO
from paddleocr import PaddleOCR
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r'C:\Users\parma\Desktop\Number Plate (Final Version)\Number plate Backend+ Website\numberplate-42ed1-firebase-adminsdk-rx812-6772a58948.json')  # Replace with path to your downloaded Firebase Admin SDK JSON file
firebase_admin.initialize_app(cred, {
    'storageBucket': 'numberplate-42ed1.appspot.com'  # Replace with your Firebase Storage bucket name
})
bucket = storage.bucket()

# Flask application setup
app = Flask(__name__)

# Initial set of predefined number plates
initial_predefined_number_plates = {"TN59AQ1515", "ABC1234", "TN59A01515", "XYZ9876", "IND5678"}
predefined_number_plates = initial_predefined_number_plates.copy()

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)  # Force use of CPU

# Load the YOLO model
number_plate_model = YOLO(r"C:\Users\parma\Desktop\Number Plate (Final Version)\Number plate Backend+ Website\best_nplate.pt")

# File path to store captured data
captured_data_file = 'captured_data.json'

# Function to load captured data from JSON file
def load_captured_data():
    if os.path.exists(captured_data_file):
        with open(captured_data_file, 'r') as file:
            loaded_data = json.load(file)
            # Extract plate numbers and add them to a set
            existing_number_plates = set(data["plate_number"] for data in loaded_data)
            # Update the predefined_number_plates set with the existing number plates
            predefined_number_plates.update(existing_number_plates)
            print("Updated predefined number plates:", predefined_number_plates)
            return loaded_data
    return []

# Function to save captured data to JSON file
def save_captured_data(data):
    with open(captured_data_file, 'w') as file:
        json.dump(data, file)

# Load captured data initially
captured_data = load_captured_data()

# Function to clean text
def clean_text(text):
    # Remove any non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '', text)

# Function to save image to Firebase Storage
def save_to_firebase(image_path, image_name):
    blob = bucket.blob(image_name)
    blob.upload_from_filename(image_path)

# Function to process cropped plate
def croped_plate(image_path):
    global predefined_number_plates
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    number_plate_results = number_plate_model.predict(img_rgb, save=False, name=image_path)
    res = ""
    unique_filename = ""

    for np_box in number_plate_results[0].boxes:
        np_cls = np_box.cls
        np_x1, np_y1, np_x2, np_y2 = np_box.xyxy[0]
        np_0x1, np_0y1, np_0x2, np_0y2 = np_x1, np_y1, np_x2, np_y2
        np_x1 = max(int(np_x1 - 350), 0)
        np_y1 = max(int(np_y1 - 350), 0)
        np_x2 = min(int(np_x2 + 350), img_rgb.shape[1])
        np_y2 = min(int(np_y2 + 350), img_rgb.shape[0])
        number_plate_image = img_rgb[int(np_y1):int(np_y2), int(np_x1):int(np_x2)]

        if number_plate_image is not None and number_plate_image.size != 0:
            number_plate_image = cv2.cvtColor(number_plate_image, cv2.COLOR_RGB2BGR)

        # OCR on the cropped number plate image
        ocr_result = ocr.ocr(number_plate_image, cls=True)

        text = ""
        for line in ocr_result:
            for word_info in line:
                word = word_info[1][0]
                cleaned_word = clean_text(word)
                if len(cleaned_word) >= 5:
                    text = cleaned_word

        if text:
            res = text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            text_color = (0, 0, 255)  # Red color (BGR)
            text_thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
            text_x = int(np_0x1 + (np_0x2 - text_size[0]) / 2)
            text_y = int(np_0y1 - 10)  # Adjust vertical position

            if text not in predefined_number_plates:
                color = (0, 255, 0)  # Green color (BGR)
            else:
                color = (0, 0, 255)  # Red color (BGR)
                # Update the set if a repeated number plate is detected
                predefined_number_plates.add(text)

            cv2.rectangle(img, (int(np_0x1), int(np_0y1)), (int(np_0x2), int(np_0y2)), color, 3)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

            # Save the processed image with a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"result_{timestamp}.jpg"
            processed_image_path = os.path.join("static", unique_filename)
            cv2.imwrite(processed_image_path, img)

            # Save the processed image to Firebase Storage
            save_to_firebase(processed_image_path, unique_filename)

            # Remove the local processed image after uploading to Firebase
            os.remove(processed_image_path)

            # Append captured data to the list
            now = datetime.now()
            new_data = {
                "plate_number": text,
                "image_url": f"gs://numberplate-42ed1.appspot.com/{unique_filename}",  # Firebase Storage path
                "date": now.strftime("%Y-%m-%d"),  # Current date
                "time": now.strftime("%H:%M:%S")  # Current time
            }
            captured_data.append(new_data)
            save_captured_data(captured_data)  # Save updated data to JSON file

    return res, unique_filename

@app.route('/')
def index():
    # Pass captured data to the HTML template, including the enumerate function
    return render_template('index.html', captured_data=captured_data, enumerate=enumerate)

@app.route('/number_plate', methods=['POST'])
def detect_license():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    # Define a directory to save the image
    upload_dir = 'uploads'

    # Create the directory if it does not exist
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the image to the directory
    image_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(image_path)

    result, unique_filename = croped_plate(image_path)
    print("result", result)
    predefined_number_plates.add(result)
    print("Predefined Number Plates:", predefined_number_plates)

    # Full URL of the processed image
    processed_image_url = f"gs://numberplate-42ed1.appspot.com/numberplate/{unique_filename}"  # Firebase Storage path

    return jsonify({"description": result, "imageUrl": processed_image_url})

@app.route('/clear_data', methods=['POST'])
def clear_data():
    global captured_data
    captured_data = []
    save_captured_data(captured_data)
    # Reset predefined_number_plates to its initial state
    global predefined_number_plates
    predefined_number_plates = initial_predefined_number_plates.copy()
    return redirect(url_for('index'))

@app.route('/remove_entry/<int:index>', methods=['POST'])
def remove_entry(index):
    global captured_data
    if 0 <= index < len(captured_data):
        captured_data.pop(index)
        save_captured_data(captured_data)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
