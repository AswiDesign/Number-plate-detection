import cv2
import numpy as np
import streamlit as st
from pytesseract import image_to_string
import pytesseract

# Streamlit app setup
st.title("Number Plate Detection")
st.text("This application detects number plates using OpenCV and displays the result.")

# Function to detect number plates
def detect_number_plate(frame, net, layer_names):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Load Haar Cascade for number plate detection
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in plates:
        # Draw rectangle around detected plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extract plate region
        plate_region = frame[y:y + h, x:x + w]
        # OCR to extract text
        plate_text = image_to_string(plate_region, config='--psm 8')
        st.text(f"Detected Plate: {plate_text.strip()}")
        cv2.putText(frame, plate_text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame

# Streamlit webcam feed
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
        frame = detect_number_plate(frame, None, None)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.write("Webcam is not running.")


# Streamlit app setup
st.title("Vehicle Number Plate Detection")
st.write("This application detects and extracts vehicle number plates using OpenCV and Tesseract OCR.")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    number_plate = None
    for contour in contours:
        # Approximate the contour
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Look for rectangular contours
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            number_plate = image[y:y + h, x:x + w]
            break

    if number_plate is not None:
        # Display the detected number plate
        st.image(number_plate, channels="BGR", caption="Detected Number Plate")

        # Apply OCR to extract text
        pytesseract.pytesseract.tesseract_cmd = r'"C:\Users\hp\AppData\Local\Programs\Python\Python313\Lib\site-packages\pytesseract\pytesseract.py"'  # Update path if necessary
        text = pytesseract.image_to_string(number_plate, config='--psm 6')
        st.write("Detected Number Plate Text:", text.strip())
    else:
        st.write("No number plate detected. Try another image.")
else:
    st.write("Please upload an image to proceed.")