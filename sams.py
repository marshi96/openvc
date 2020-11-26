from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
from PIL import Image
import pytesseract
from xml.dom import minidom
import os, sys
import csv
from pathlib import Path

argument_data = sys.argv

# for i in len(2):
if (len(argument_data) < 3):
    print('Insert date as commandline argument')
    sys.exit(0)
# Open and resize images

image = cv2.imread(argument_data[1])
#image = cv2.imread("3.jpeg")

image = imutils.resize(image, width=612, height=800)

image2 = Image.open(argument_data[1])
#image2 = cv2.imread("3.jpeg")

image2 = image2.resize((612, 800), Image.ANTIALIAS)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 50, 200, 255)

# initializing variables
count = 0
crop_sign = None
crop_index = None

# to detect lines by using canny edges
minLineLength = 2
maxLineGap = 4
lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 50, minLineLength, maxLineGap, 8)


def takeSecond(elem):
    return elem[0][1]


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

displayCnt = None

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        displayCnt = approx
        break

warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

# create testing folder on following path
if not os.path.exists("testing"):
    os.mkdir('testing')
doc = minidom.parse(argument_data[2])
present_name = list()
image_data = np.array([])
# go through the range of signs on the image
for i in range(15, 120, 17):

    crop_sign = output[i:i + 19, 340:340 + 92]
    dst = cv2.fastNlMeansDenoisingColored(crop_sign, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image to covert to binary and resize
    thresh_sign = cv2.threshold(blurred, 195, 255, cv2.THRESH_BINARY)[1]
    image_sign = imutils.resize(thresh_sign, width=612, height=800)

    cv2.imshow("cropped_sign", image_sign)

    cv2.imwrite("testing/testing.jpeg", thresh_sign)
    imageTest = Image.open("testing/testing.jpeg")

    # count the white pixels by countNonZero() function
    data = np.array(imageTest.getdata(),
                    np.uint8)
    nzCount = cv2.countNonZero(data)
    print(nzCount)

    count += 1

    # if number of white pixels less than thousand said to be signed
    if 10 < nzCount < 1000:
        # show cropped number on the sheet
        crop_index = output[i:i + 19, 1:1 + 50]
        cv2.imshow("cropped_index", crop_index)
        print("count: " + str(count))

        # change these xml path to your path
        sName = doc.getElementsByTagName('studentName')

        name = sName[count - 1].firstChild.data
        present_name.append(name)

        # store the image of sign with the name on created path
        path = "testing/" + name + ".jpeg"

        image_data = np.append(image_data, crop_sign)

        # show the cropped sign on screen
        cv2.imwrite(path, crop_sign)

cv2.imshow('image', output)

print(count)

csv.register_dialect('myDialect', delimiter='/', quoting=csv.QUOTE_NONE)
students_name_id = []
attendance = []
my_file = Path("Attendance.csv")

location = Path(argument_data[1]).stem
# print(location)
date = location
if my_file.is_file():
    with open('Attendance.csv', newline='') as myFile:
        spamreader = csv.reader(myFile, dialect='myDialect')
        print(spamreader)
        reader = next(spamreader)
        csv_writer = csv.writer(myFile)
        first_line = [s.strip() for s in str(reader[0]).split(',')]
        print(first_line) #

        for i in first_line: #
           if ('23.04.2017' in first_line):
            print('Already marked the attendance')
           else:
            first_line.append(date)
            print(first_line) #
        for row in spamreader:
            student_data = row[0].split(",")
            student_data.append('0')
            for c in present_name:
                if (student_data[1] == c):
                    student_data[-1] = 1
            attendance.append(student_data)

    with open('Attendance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_line)
        writer.writerows(attendance)
else:
    row_list = ["Name", "id", date[1]]
    s_name = doc.getElementsByTagName('studentName')
    s_id = doc.getElementsByTagName('studentNumber')

    for n, i in zip(s_name, s_id):
        students_name_id.append([i.firstChild.data, n.firstChild.data, '0'])
    for i in students_name_id:
        for c in present_name:
            if (i[1] == c):
                i[2] = '1'

    with open('Attendance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_list)
        writer.writerows(students_name_id)
        print(students_name_id) #
