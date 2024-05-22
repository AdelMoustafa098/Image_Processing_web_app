import numpy as np




class ImageProcessor():
    def __init__(self, image):
        self.image = image
        self.orignal_image = image

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


    def convert_to_gry(self):
        """
            This method convert an RGB image to gray scale image doing the follwoing:

                1- extract the rgb channels from the image.
                2- multiply each channel by a constant and sum the result.
                3- convert the result from step 2 to a sutible data type.

            Arguments: None
            Returns: None

        """
        
        b, g, r = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        self.image = gray.astype(np.uint8)


    def add_salt_pepper_noise(self):
        pass
    
