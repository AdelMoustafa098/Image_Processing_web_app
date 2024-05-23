import cv2
from image_processor import ImageProcessor


img = cv2.imread("Test Images/2_50.jpg", cv2.IMREAD_COLOR)


ip = ImageProcessor(img)
# ip.convert_to_gry()
# ip.reset_image()
ip.add_salt_pepper_noise()
x = ip.get_image()
cv2.imshow("image", x)
cv2.waitKey(0)
