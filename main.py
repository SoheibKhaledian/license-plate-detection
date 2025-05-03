import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('license_plate_detector.pt')
image = cv2.imread('car.jpeg')

results = model(image)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        plate_crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(thresh, config='--psm 7')
        print(f"Detected Text {i+1}: {text.strip()}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, text.strip(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow('Detected Plates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
