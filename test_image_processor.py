from image_processor import ImageProcessor
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
    np.testing.assert_array_equal(reset_image, img)


def test_convert_to_gray():

    rgb_image = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8
    )

    # Expected grayscale values (calculated using the weights 0.2989, 0.5870, 0.1140)
    expected_gray_image = np.array([[29, 149], [76, 254]], dtype=np.uint8)

    ip = ImageProcessor(rgb_image)
    ip.convert_to_gry()
    gray_image = ip.get_image()

    np.testing.assert_allclose(gray_image, expected_gray_image, atol=1)


def test_add_salt_pepper_noise():

    sample_image = np.full((100, 100), 128, dtype=np.uint8)
    ip = ImageProcessor(sample_image)
    ip.add_salt_pepper_noise()

    # Check if the image is still the same shape
    assert ip.image.shape == sample_image.shape

    # Check if noise has been added by counting non-128 values
    unique, counts = np.unique(ip.get_image(), return_counts=True)
    counts_dict = dict(zip(unique, counts))

    # Ensure there are pixels set to 0 and 255
    assert 0 in counts_dict
    assert 255 in counts_dict

    # Check if there are at least 100 noisy pixels
    noise_count = counts_dict.get(0, 0) + counts_dict.get(255, 0)
    assert noise_count >= 100
