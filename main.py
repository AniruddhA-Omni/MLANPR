import cv2
import pytesseract
from googletrans import Translator
import time
import imutils
import numpy as np


# Initialisation
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
translator = Translator()

# Reading Image
img = cv2.imread('car_test2.jpg')

############################################## Number plate extraction #################################################
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 5, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
cv2.imwrite("new_en.jpeg", cropped_image)
######################################################## Image Enhancing ###############################################

# Converting to grayscale
# gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Applying Gaussian Blur to image
blur = cv2.GaussianBlur(cropped_image, (3, 3), 0)

# Thresholding
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise and invert image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening

############################################### Character segmentation #################################################

while True:
    op = int(input("Options: \n1. Bengali \n2. Tamil \n3. Telegu \n4. Hindi \n5. English \nEnter option no.:"))
    if op == 1:
        langp, langg = "ben", "bn"
        break
    elif op == 2:
        langp, langg = "tam", "ta"
        break
    elif op == 3:
        langp, langg = "tel", "te"
        break
    elif op == 4:
        langp, langg = "hin", "hi"
        break
    elif op == 5:
        langp, langg = "eng", "en"
        break
    else:
        print("Wrong option entered!!")

st = time.time()    # checking
text = pytesseract.image_to_string(invert, lang=langp, config='--psm 6')
print("Detected Text:", text)

text_s = "".join(text.split())
# print(text_s) # segmented characters
res_s = ""
for i in text_s:
    result = translator.translate(i, src=langg, dest='en')
    # if len(str(result.text)) == 1:
    res_s += str(result.text)
print("Translated text:", res_s)
et = time.time()    # checking
print("Time taken:", et-st)
