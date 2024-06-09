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
    _ = ip.convert_to_gray(img)
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
    gray_image = ip.convert_to_gray(rgb_image)

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


def test_add_gaussian_noise():

    # Create a simple 2x2 grayscale image
    gray_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)

    # Create an instance of ImageProcessor
    processor = ImageProcessor(gray_image)

    # Mock the numpy random normal function to produce a predictable output
    def mock_normal(mean, std, size):
        return np.array([[5, -5], [10, -10]], dtype=np.float64)

    # Save the original np.random.normal function
    original_normal = np.random.normal

    # Replace the np.random.normal function with the mock
    np.random.normal = mock_normal

    try:
        # Add Gaussian noise
        processor.add_gaussian_noise()

        # Get the noisy image
        noisy_image = processor.get_image()

        # Expected noisy image
        expected_noisy_image = np.array([[105, 145], [210, 240]], dtype=np.uint8)

        # Print both images for debugging purposes
        print("Expected noisy image:\n", expected_noisy_image)
        print("Actual noisy image:\n", noisy_image)

        # Assert the arrays are equal
        np.testing.assert_array_equal(noisy_image, expected_noisy_image)
    finally:
        # Restore the original np.random.normal function
        np.random.normal = original_normal


def test_add_uniform_noise():
    # Create a simple 2x2 grayscale image
    gray_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)

    # Create an instance of ImageProcessor
    processor = ImageProcessor(gray_image)

    # Mock the numpy random uniform function to produce a predictable output
    def mock_uniform(low, high, size):
        return np.array([[10, -10], [5, -5]], dtype=np.float64)

    # Save the original np.random.uniform function
    original_uniform = np.random.uniform

    # Replace the np.random.uniform function with the mock
    np.random.uniform = mock_uniform

    try:
        # Add uniform noise
        processor.add_uniform_noise()

        # Get the noisy image
        noisy_image = processor.get_image()

        # Expected noisy image
        expected_noisy_image = np.array([[110, 140], [205, 245]], dtype=np.uint8)

        # Print both images for debugging purposes
        print("Expected noisy image:\n", expected_noisy_image)
        print("Actual noisy image:\n", noisy_image)

        # Assert the arrays are equal
        np.testing.assert_array_equal(noisy_image, expected_noisy_image)
    finally:
        # Restore the original np.random.uniform function
        np.random.uniform = original_uniform
