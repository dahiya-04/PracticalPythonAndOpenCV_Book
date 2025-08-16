import cv2

from google.colab.patches import cv2_imshow
# Load the image and show some basic information on it
image = cv2.imread('/content/Screenshot 2025-08-16 142753.png')
print("Width: {} pixels".format(image.shape[1]))
print("Height: {} pixels".format(image.shape[0]))
print("Channels: {}".format(image.shape[2]))

#Lines 6-7 examine the dimensions of the image

# Show the image and wait for a keypress
cv2_imshow(image)
cv2.waitKey(0)
# Finally, Lines 14 handle displaying the actual
# image on our screen
# Save the image
cv2.imwrite("saved_image.jpg", image)
