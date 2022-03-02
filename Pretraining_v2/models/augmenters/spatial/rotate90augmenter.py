"""
This file contains a class for augmenting patches from whole slide images with rotating by multiples of 90 degrees.
"""

from . import spatialaugmenterbase as dptspatialaugmenterbase


import numpy as np

#----------------------------------------------------------------------------------------------------

class Rotate90Augmenter(dptspatialaugmenterbase.SpatialAugmenterBase):
    """Rotate patch by 90, 180 or 270 degrees."""

    def __init__(self, k_list):
        """
        Initialize the object.

        Args:
            k_list (list): List of 90 degree rotation repetition times. Example: k_list = [0, 1, 2, 3] for 0, 90,
                180 and 270 degrees.

        Raises:
            InvalidRotationRepetitionListError: The list for 90 degree rotation repetition is invalid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='rotate_90')

        # Initialize members.
        #
        self.__k_list = []  # List of rotation repetitions to use.
        self.__k = None     # Current repetition number to use.

        # Save configuration.
        #
        self.__setklist(k_list=k_list)

    def __setklist(self, k_list):
        """
        Set the rotation repetition times list.

        Args:
            k_list (list): List of 90 degree rotation repetition times.

        Raises:
            InvalidRotationRepetitionListError: The list for 90 degree rotation repetition is invalid.
        """

        # Check the list.
        #
        if len(k_list) < 1 or any(isinstance(k_item, float) and not float.is_integer(k_item) for k_item in k_list):
            raise Exception("InvalidRotationRepetitionListError(k_list)")

        # Store the setting.
        #
        self.__k_list = [int(k_item) % 4 for k_item in k_list]
        self.__k = self.__k_list[0]

    def transform(self, patch):
        """
        Rotate the patch with multiple of 90 degrees.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Rotate the patch.
        #
        patch_transformed = np.rot90(m=patch, k=self.__k, axes=(1, 2))

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize the K.
        #
        self.__k = np.random.choice(a=self.__k_list, size=None)
