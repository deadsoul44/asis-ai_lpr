# remove warning message
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required libraries
import cv2
import numpy as np
from local_utils import detect_lp
from os.path import splitext
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import re

test_images_folder = "img"
cwd = os.getcwd()
test_images_folder = cwd + "\\" + test_images_folder + "\\*"


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(image_path, Dmax=100, Dmin=100):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.1)
    return vehicle, LpImg, cor


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")


# pre-processing input images and pedict with model
def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


number_of_images = 0
correct_predictions = 0
# Loop for all images
for img in glob.glob(test_images_folder):
    print(img)
    number_of_images += 1
    actual_plate = img.split("\\")[-1].split(".")[0]

    for i in range(120, 20, -8):
        final_string = ''
        try:
            vehicle, LpImg, cor = get_plate(img, i, i)

            # Part 2: Segmenting license characters
            if len(LpImg):  # check if there is at least one license image
                # Scales, calculates absolute values, and converts the result to 8-bit.
                plate_image = cv2.convertScaleAbs(LpImg[0], alpha=255.0)

                # convert to grayscale and blur the image
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (1, 1), 0)

                # Applied inversed thresh_binary
                binary = cv2.threshold(blur, 18, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            cont, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append charater image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 30, 60

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1 <= ratio <= 15:  # Only select contour with defined ratio
                    if h / plate_image.shape[
                        0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Sperate number and gibe prediction
                        curr_num = binary[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)

            for j, character in enumerate(crop_characters):
                title = np.array2string(predict_from_model(character, model, labels))
                final_string += title.strip("'[]")

            final_string = re.sub("(?<=\d)[A-Z]{1}(?=\d)", "0", final_string)

            if not (final_string[0].isdigit()):
                s = list(final_string)
                if s[0] == "O":
                    s[0] = str(0)
                else:
                    del s[0]
                final_string = "".join(s)

            if not (final_string[1].isdigit()):
                s = list(final_string)
                s.insert(0, "0")
                final_string = "".join(s)

        except:
            continue

        try:
            if len(crop_characters) > 6:
                break
        except:
            continue

    if final_string == actual_plate:
        correct_predictions += 1

    print(actual_plate, final_string)

accuracy = correct_predictions / number_of_images
print("correct_predictions: ", str(correct_predictions))
print("number_of_images: ", str(number_of_images))
print("accuracy: ", str(accuracy))
