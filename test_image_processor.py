from image_processer import ImageProcessor
import cv2
import numpy as np
img = cv2.imread("Test Images/2_50.jpg", cv2.IMREAD_COLOR)

def test_get_image():

    ip = ImageProcessor(img)
    test_img = ip.get_image()
    np.testing.assert_array_equal(test_img, img)


def test_reset_image():

    ip = ImageProcessor(img)
    ip.convert_to_gry()
    ip.reset_image()
    reset_image = ip.get_image()    
    np.testing.assert_array_equal(reset_image,img)




def test_convert_to_gray():
    # Create a simple 2x2 RGB image
    rgb_image = np.array([[[255, 0, 0], [0, 255, 0]],
                          [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    
    # Expected grayscale values (calculated using the weights 0.2989, 0.5870, 0.1140)
    expected_gray_image = np.array([[29, 149],
                                    [76, 254]], dtype=np.uint8)
    
    
    ip = ImageProcessor(rgb_image)
    ip.convert_to_gry()
    gray_image = ip.get_image()   
    

    np.testing.assert_allclose(gray_image, expected_gray_image, atol=1)
