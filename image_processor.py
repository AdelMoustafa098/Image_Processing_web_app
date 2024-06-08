import numpy as np
from scipy import signal


class ImageProcessor:
    """
    This class is used for image processing
    original
    """

    def __init__(self, image):
        self.image = image
        self.original_image = image

    def get_image(self):
        """
        This method returns the image

        Arguments: None
        Returns: image (numpy array)
        """
        return self.image

    def reset_image(self):
        """
        This method resets the image to the original input image removing any processing done on it

        Arguments: None
        Returns: None
        """
        self.image = self.original_image

    def is_gray(self):
        """
        This method checks if the image is gray scale.

        Arguments: None
        Returns: bool
        """
        return len(self.image.shape) == 2

    def convert_to_gray(self):
        """
        This method convert an RGB image to gray scale image doing the following:

            1- extract the rgb channels from the image.
            2- multiply each channel by a constant and sum the result.
            3- convert the result from step 2 to a suitable data type.

        Arguments: None
        Returns: None

        """

        b, g, r = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        self.image = gray.astype(np.uint8)

    def add_salt_pepper_noise(self):
        """
        This method add salt and pepper noise to the gray image

        Arguments: None
        Returns: None
        """
        # make sure image is gray
        if not self.is_gray():
            self.convert_to_gray()

        row, col = self.image.shape
        selected_pixel = np.random.randint(100, 5000)

        # Generate coordinates for white (salt) noise
        white_coords = (
            np.random.randint(0, row, selected_pixel),
            np.random.randint(0, col, selected_pixel),
        )
        self.image[white_coords] = 255

        # Generate coordinates for black (pepper) noise
        black_coords = (
            np.random.randint(0, row, selected_pixel),
            np.random.randint(0, col, selected_pixel),
        )
        self.image[black_coords] = 0

    def add_gaussian_noise(self):
        """
        This method adds a gaussian noise to a gray image with zero mean and 15 standard deviation

        Arguments: None
        Returns: None
        """

        # make sure image is gray
        if not self.is_gray():
            self.convert_to_gray()

        row, col = self.image.shape
        # create gaussian noise
        mean = 0.0
        std = 15.0
        noise = np.random.normal(mean, std, size=(row, col))

        # apply noise
        self.image = np.add(self.get_image(), noise)
        self.image = self.image.astype(np.uint8)

    def add_uniform_noise(self):
        """
        This method adds a uniform noise to a gray image

        Arguments: None
        Returns: None
        """
        # make sure image is gray
        if not self.is_gray():
            self.convert_to_gray()

        # create uniform image
        row, col = self.image.shape
        noise = np.random.uniform(-20, 20, size=(row, col))

        # apply noise
        self.image = np.add(self.image, noise)
        self.image = self.image.astype(np.uint8)

    def apply_mask(self, mask: np.array):
        """
        This method take a mask and apply it to the instance image

        Arguments:
            mask (numpy.array): mask to be applied

        Returns:None
        """
        if not self.is_gray():
            self.convert_to_gray()
        self.image = signal.convolve2d(self.get_image(), mask)
        self.image = self.image.astype(np.uint8)

    def avg_filter(self):
        """
        This method applies a 3x3 average filter on a gray image

        Arguments: None
        Returns: None
        """
        mask = np.ones([3, 3], dtype=int)
        mask = mask / 9
        self.apply_mask(mask)

    def gaussian_filter(self):
        """
        This method applies a Gaussian filter with a sigma of 2.6 and a kernel size of (9, 9).

        Arguments: None
        Returns: None
        """

        # Define the standard deviation (sigma) and the size of the Gaussian kernel
        sigma = 2.6
        kernel_size = (9, 9)

        # Calculate the center coordinates of the kernel
        center_y, center_x = [(size - 1.0) / 2.0 for size in kernel_size]

        # Generate a grid of (y, x) coordinates
        y, x = np.ogrid[-center_y : center_y + 1, -center_x : center_x + 1]

        # Compute the Gaussian function
        gaussian_kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

        # Set very small values to zero
        gaussian_kernel[
            gaussian_kernel
            < np.finfo(gaussian_kernel.dtype).eps * gaussian_kernel.max()
        ] = 0

        # Normalize the kernel so that its sum is 1
        kernel_sum = gaussian_kernel.sum()
        if kernel_sum != 0:
            gaussian_kernel /= kernel_sum

        # Apply the Gaussian filter to the image
        return self.apply_mask(gaussian_kernel)
