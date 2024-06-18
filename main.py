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

app = Flask(__name__)

# Initial set of predefined number plates
initial_predefined_number_plates = {"TN59AQ1515", "ABC1234", "TN59A01515", "XYZ9876", "IND5678"}
predefined_number_plates = initial_predefined_number_plates.copy()

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)  # Force use of CPU

# Load the YOLO model
number_plate_model = YOLO(r"C:\Users\parma\Desktop\Number Plate (Final Version)\Number plate Backend+ Website\best_nplate.pt")

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r'C:\Users\parma\Desktop\Number Plate (Final Version)\Number plate Backend+ Website\numberplate-42ed1-firebase-adminsdk-rx812-6772a58948.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'numberplate-42ed1.appspot.com'
})
bucket = storage.bucket()

# File path to store captured data
captured_data_file = 'captured_data.json'

# Function to load captured data from JSON file
def load_captured_data():
    if os.path.exists(captured_data_file):
        with open(captured_data_file, 'r') as file:
            loaded_data = json.load(file)
            existing_number_plates = {data["plate_number"] for data in loaded_data}
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
    return re.sub(r'[^a-zA-Z0-9]', '', text)

# Function to upload image to Firebase Storage
def upload_to_firebase(local_path, firebase_path):
    blob = bucket.blob(firebase_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url

# Function to process cropped plate
def croped_plate(image_path):
    global predefined_number_plates
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    number_plate_results = number_plate_model.predict(img_rgb, save=False)
    res = ""
    unique_filename = ""

    for np_box in number_plate_results[0].boxes:
        np_x1, np_y1, np_x2, np_y2 = map(int, np_box.xyxy[0])
        np_x1, np_y1, np_x2, np_y2 = max(np_x1 - 350, 0), max(np_y1 - 350, 0), min(np_x2 + 350, img_rgb.shape[1]), min(np_y2 + 350, img_rgb.shape[0])
        number_plate_image = img_rgb[np_y1:np_y2, np_x1:np_x2]

        if number_plate_image.size != 0:
            number_plate_image = cv2.cvtColor(number_plate_image, cv2.COLOR_RGB2BGR)
            ocr_result = ocr.ocr(number_plate_image, cls=True)

            for line in ocr_result:
                for word_info in line:
                    word = word_info[1][0]
                    cleaned_word = clean_text(word)
                    if len(cleaned_word) >= 5:
                        text = cleaned_word
                        break

            if text:
                res = text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                text_color = (0, 0, 255)  # Red color (BGR)
                text_thickness = 3
                text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
                text_x = np_x1 + (np_x2 - text_size[0]) // 2
                text_y = np_y1 - 10  # Adjust vertical position

                color = (0, 255, 0) if text not in predefined_number_plates else (0, 0, 255)
                if text in predefined_number_plates:
                    predefined_number_plates.add(text)

                cv2.rectangle(img, (np_x1, np_y1), (np_x2, np_y2), color, 3)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

                # Save the processed image with a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"result_{timestamp}.jpg"
                processed_image_path = os.path.join("static", unique_filename)
                cv2.imwrite(processed_image_path, img)

                # Upload the processed image to Firebase Storage
                firebase_filename = f"numberplate/{text}.jpg"
                firebase_url = upload_to_firebase(processed_image_path, firebase_filename)

                # Append captured data to the list
                now = datetime.now()
                new_data = {
                    "plate_number": text,
                    "image_url": f"/static/{unique_filename}",  # Relative path to the processed image
                    "firebase_url": firebase_url,  # URL of the image in Firebase Storage
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
    print("Result:", result)
    predefined_number_plates.add(result)
    print("Predefined Number Plates:", predefined_number_plates)

    # Full URL of the processed image
    processed_image_url = url_for('static', filename=unique_filename, _external=True)

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

