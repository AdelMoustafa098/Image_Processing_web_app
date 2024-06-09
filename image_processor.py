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

    def is_gray(self, image: np.array):
        """
        This method checks if the image is gray scale.

        Arguments:
            image (numpy.array): image which to be cheked
        Returns: bool
        """
        return len(image.shape) == 2

    def convert_to_gray(self, image: np.array):
        """
        This method convert an RGB image to gray scale image doing the following:

            1- extract the rgb channels from the image.
            2- multiply each channel by a constant and sum the result.
            3- convert the result from step 2 to a suitable data type.

        Arguments:
            image (numpy.array): image to be converted to gray
        Returns:
            gray_image (numpy.array): gray image

        """

        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        self.image = gray.astype(np.uint8)
        gray_image = gray.astype(np.uint8)
        return gray_image

    def add_salt_pepper_noise(self):
        """
        This method add salt and pepper noise to the gray image

        Arguments: None
        Returns: None
        """
        # make sure image is gray
        if not self.is_gray(self.image):
            self.image = self.convert_to_gray(self.image)

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
        if not self.is_gray(self.image):
            self.image = self.convert_to_gray(self.image)

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
        if not self.is_gray(self.image):
            self.image = self.convert_to_gray(self.image)

        # create uniform image
        row, col = self.image.shape
        noise = np.random.uniform(-20, 20, size=(row, col))

        # apply noise
        self.image = np.add(self.image, noise)
        self.image = self.image.astype(np.uint8)

    def apply_mask(self, image: np.array, mask: np.array):
        """
        This method take a mask and apply it to the instance image

        Arguments:
            image (numpy.array): image which the mask is going to be applied on
            mask (numpy.array): mask to be applied

        Returns:
            masked_image (numpy.array): image after mask application
        """
        if not self.is_gray(image):
            image = self.convert_to_gray(image)
        image = signal.convolve2d(image, mask)
        masked_image = image.astype(np.uint8)
        return masked_image

    def avg_filter(self):
        """
        This method applies a 3x3 average filter on a gray image

        Arguments: None
        Returns: None
        """
        mask = np.ones([3, 3], dtype=int)
        mask = mask / 9
        self.image = self.apply_mask(self.image, mask)

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
        self.image = self.apply_mask(self.image, gaussian_kernel)

    def sobel_edge(self):
        """
        This method applies a sobel edge detector.

        Arguments: None
        Returns: None
        """
        # Apply Gaussian filter to smooth the image
        self.gaussian_filter()

        # Define Sobel kernels
        kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        ky = kx.T

        # Compute gradients using convolution
        Ix = signal.convolve2d(self.get_image(), kx, mode="same", boundary="symm")
        Iy = signal.convolve2d(self.get_image(), ky, mode="same", boundary="symm")

        # Calculate magnitude and direction of the gradient
        magnitude = np.hypot(Ix, Iy)
        direction = np.arctan2(Iy, Ix)

        # Convert direction to degrees and shift range from [-180, 180] to [0, 360]
        direction = np.rad2deg(direction) + 180

        # Convert to uint8 type
        self.image = np.clip(magnitude, 0, 255).astype(np.uint8)

    def roberts_edge(self):
        """
        This method applies a roberts edge detector.

        Arguments: None
        Returns: None
        """

        mask_x = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

        mask_y = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

        self.gaussian_filter()
        mask_x_dirc = self.apply_mask(self.image, mask_x)
        mask_y_dirc = self.apply_mask(self.image, mask_y)

        gradient = np.sqrt(np.square(mask_x_dirc) + np.square(mask_y_dirc))
        self.image = np.uint8((gradient * 255.0) / gradient.max())
