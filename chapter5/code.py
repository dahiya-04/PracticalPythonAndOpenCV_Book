#translation 
# Translation is the shifting of an image along the x and y
# axis. Using translation, we can shift an image up, down,
# left, or right, along with any combination of the above

import numpy as np
import imutils
import cv2


# Load the image and show it
image = cv2.imread('/content/Screenshot 2025-08-16 142753.png')
cv2_imshow(image)

# Translate the image 25 pixels to the right and 50 pixels down
matrix = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
cv2_imshow(shifted)

# Shift the image 50 pixels to the left and 90 pixels up
M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2_imshow(shifted)

#Rotation
import cv2
import imutils

image = cv2.imread('/content/Screenshot 2025-08-16 142753.png')
cv2_imshow(image)

(h,w)= image.shape[:2]
center=(w/2,h/2)

M = cv2.getRotationMatrix2D(center, 100, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2_imshow(rotated)

# resize

image = cv2.imread('/content/Screenshot 2025-08-16 142753.png')
cv2_imshow(image)

# # Calculate the ratio based on the width to keep the aspect ratio
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

# Perform the actual resizing of the image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2_imshow(resized)

# Calculate the ratio based on height to keep the aspect ratio
r = 50.0 / image.shape[0]
dim = (int(image.shape[1] * r), 50)

# Perform the resizing
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2_imshow(resized)
cv2.waitKey(0)

#cropping

cv2_imshow(image[0:300,0:300])
cv2.waitKey(0)


#arthimatics
cv2_imshow(image)

# Demonstrate 'unsigned integers' for addition and subtraction using OpenCV
print("Max of 255: {}".format(cv2.add(np.uint8([200]), np.uint8([100]))))
print("Min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))

# Demonstrate 'unsigned intergers' for addition and subtraction using NumPy
print("wrap around: {}".format(np.uint8([200]) + np.uint8([100])))
print("wrap around: {}".format(np.uint8([50]) - np.uint8([100])))

# Increase the intensity of all pixels in our image by 100 (to make it brighter)
mask = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, mask)
cv2_imshow(added)


#merge and add
(B, G, R) = cv2.split(image)

# Show each channel individually
cv2_imshow( R)
cv2_imshow(G)
cv2_imshow(B)
cv2.waitKey(0)

# Merge the image back together again
merged = cv2.merge([B, G, R])
cv2_imshow(merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Now, let's visualize each channel in color
zeros = np.zeros(image.shape[:2], dtype="uint8")
cv2_imshow(cv2.merge([zeros, zeros, R]))
cv2_imshow(cv2.merge([zeros, G, zeros]))
cv2_imshow(cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
