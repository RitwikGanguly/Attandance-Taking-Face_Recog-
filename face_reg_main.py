# WELCOME TO FACE RECOGNITION USING PYTHON
# AUTHOR - RITWIK GANGULY
# This is the Testing page of the face recognition using opencv library

# The next page consist of 'AUTO ATTENDANCE SYSTEM USING FACE RECOGNITION
# THROUGH WEBCAM'








# importing libraries
import cv2
import numpy as np
import face_recognition

imgvk = face_recognition.load_image_file("images/ritwik.jpg")
img_tr = cv2.cvtColor(imgvk, cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file("images/soumya-999.jpg")
img_ts = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

# -----------FOR THE TRAIN IMAGE---------------
# detecting the face/ as sending a single image will take the first element of this
face_loc = face_recognition.face_locations(img_tr)[0]

# to check the face corner onto the face
print(face_loc)

# encode the face we have detected
encode_vk = face_recognition.face_encodings(img_tr)[0]
cv2.rectangle(img_tr, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), color=(255, 0, 255), thickness=2)


# -----------FOR THE TEST IMAGE---------------

# encode the face we have detected

face_loc_test = face_recognition.face_locations(img_ts)[0]
encode_test = face_recognition.face_encodings(img_ts)[0]
cv2.rectangle(img_ts, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), color=(255, 200, 255), thickness=2)
# ------------COMPARING THE FACES------------------

# compare and find the distance between them
# will be used linear svm to training the model
# have only two images one for train and another for test, will use those images
results = face_recognition.compare_faces([encode_vk], encode_test)

# face distance to check the similarities
# the lower the distance the better the match is.
facedis = face_recognition.face_distance([encode_vk], encode_test)

# print the boolean match/not-matched and distance
# print(results, facedis)

# display the result in actual result image
if facedis < 0.43:
    print(results, facedis)
    cv2.putText(img_ts, "{} {}".format(results, round(facedis[0], 2)), (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 240), 2)



# to show the images

    cv2.imshow('virat', img_tr)
    cv2.imshow('virat test', img_ts)

    cv2.waitKey(0)

else:
    print(facedis, "not totally matched")