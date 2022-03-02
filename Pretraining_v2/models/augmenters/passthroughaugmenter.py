"""
This file contains a pass-through augmentation cass.
"""

from . import augmenterbase as dptaugmenterbase

#----------------------------------------------------------------------------------------------------

class PassThroughAugmenter(dptaugmenterbase.AugmenterBase):
    """Pass through augmenter that does noting."""

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__(keyword='pass_through')

    def transform(self, patch):
        """
        Return the given patch without transformation.

        Args:
            patch (np.ndarray): Patch to return.

        Returns:
            np.ndarray: The patch.
        """

        return patch
