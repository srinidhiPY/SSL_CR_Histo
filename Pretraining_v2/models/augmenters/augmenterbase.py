"""
This file contains base class for augmenting patches from whole slide images.
"""

#----------------------------------------------------------------------------------------------------

class AugmenterBase(object):
    """Base class for patch augmentation."""

    def __init__(self, keyword):
        """
        Initialize the object.

        Args:
            keyword (str): Short name for the transformation.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__keyword = keyword

    @property
    def keyword(self):
        """
        Get the keyword for the augmenter.

        Returns:
            str: Keyword.
        """

        return self.__keyword

    def shapes(self, target_shapes):
        """
        Calculate the required shape of the input to achieve the target output shape.

        Args:
            target_shapes (dict): Target output shape per level.

        Returns:
            (dict): Required input shape per level.
        """

        # By default the output shapes match the input shapes.
        #
        return target_shapes

    def transform(self, patch):
        """
        Transform the given patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        pass

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        pass
