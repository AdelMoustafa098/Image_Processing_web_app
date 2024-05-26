import numpy as np
from scipy import signal


class ImageProcessor:
    """
    This class is used for image processing
    """

    def __init__(self, image):
        self.image = image
        self.orignal_image = image
        self.gray_image = False

    def get_image(self):
        """
        This method returns the image

        Arguments: None
        Returns: image (numpay array)
        """
        return self.image

    def reset_image(self):
        """
        This method resets the image to the original input image removing any processing done on it

        Arguments: None
        Returns: None
        """
        self.image = self.orignal_image

    def is_gray(self):
        """
        This method checks if the image is gray scale.

        Arguments: None
        Returns: bool
        """
        return len(self.image.shape) == 2

    def convert_to_gry(self):
        """
        This method convert an RGB image to gray scale image doing the follwoing:

            1- extract the rgb channels from the image.
            2- multiply each channel by a constant and sum the result.
            3- convert the result from step 2 to a sutible data type.

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
            self.convert_to_gry()

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

    def add_gussian_noise(self):
        """
        This method adds a gussian noise to a gray image with zero mean and 15 sandard deviation

        Arguments: None
        Returns: None
        """

        # make sure image is gray
        if not self.is_gray():
            self.convert_to_gry()

        row, col = self.image.shape
        # create gussian noise
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
            self.convert_to_gry()

        # create uniform image
        row, col = self.image.shape
        noise = np.random.uniform(-20, 20, size=(row, col))

        # apply noise
        self.image = np.add(self.image, noise)
        self.image = self.image.astype(np.uint8)

    def apply_mask(self, mask: np.array):
        """
        This method take a mask and apply it to the instace image

        Arguments:
            mask (numpay.array): mask to be appleid

        Returns:None
        """
        
        self.image = signal.convolve2d(np.reshape(self.image, (-1, 2)), mask)
        self.image = self.image.astype(np.uint8)

    def avg_filter(self):
        """
        This method applys a 3x3 average filter on a gray image

        Arguments: None
        Returns: None
        """
        mask = np.ones([3, 3], dtype=int)
        mask = mask / 9
        self.apply_mask(mask)
