import cv2
import pytesseract
from googletrans import Translator
import time


# Initialisation
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
translator = Translator()

# Reading Image
img = cv2.imread('test_img.jpg')

############################################## Number plate extraction #################################################

######################################################## Image Enhancing ###############################################

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying Gaussian Blur to image
blur = cv2.GaussianBlur(gray, (3, 3), 0)

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
