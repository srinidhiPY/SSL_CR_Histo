"""
This file contains a class for augmenting patches from whole slide images by applying color correction in HSB color space.
"""

from . import coloraugmenterbase as dptcoloraugmenterbase

import skimage.color
import numpy as np

#----------------------------------------------------------------------------------------------------

class HsbColorAugmenter(dptcoloraugmenterbase.ColorAugmenterBase):
    """Apply color correction in HSB color space on the RGB patch."""

    def __init__(self, hue_sigma_range, saturation_sigma_range, brightness_sigma_range):
        """
        Initialize the object.

        Args:
            hue_sigma_range (tuple, None): Adjustment range for the Hue channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.5, 0.5).
            saturation_sigma_range (tuple, None): Adjustment range for the Saturation channel from the [-1.0, 1.0] range where 0.0 means no change.
            brightness_sigma_range (tuple, None): Adjustment range for the Brightness channel from the [-1.0, 1.0] range where 0.0 means no change.

        Raises:
            InvalidHueSigmaRangeError: The sigma range for Hue channel adjustment is not valid.
            InvalidSaturationSigmaRangeError: The sigma range for Saturation channel adjustment is not valid.
            InvalidBrightnessSigmaRangeError: The sigma range for Brightness channel adjustment is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='hsb_color')

        # Initialize members.
        #
        self.__sigma_ranges = None  # Configured sigma ranges for H, S, and B channels.
        self.__sigmas = None        # Randomized sigmas for H, S, and B channels.

        # Save configuration.
        #
        self.__setsigmaranges(hue_sigma_range=hue_sigma_range, saturation_sigma_range=saturation_sigma_range, brightness_sigma_range=brightness_sigma_range)

    def __setsigmaranges(self, hue_sigma_range, saturation_sigma_range, brightness_sigma_range):
        """
        Set the sigma ranges.

        Args:
            hue_sigma_range (tuple, None): Adjustment range for the Hue channel.
            saturation_sigma_range (tuple, None): Adjustment range for the Saturation channel.
            brightness_sigma_range (tuple, None): Adjustment range for the Brightness channel.

        Raises:
            InvalidHueSigmaRangeError: The sigma range for Hue channel adjustment is not valid.
            InvalidSaturationSigmaRangeError: The sigma range for Saturation channel adjustment is not valid.
            InvalidBrightnessSigmaRangeError: The sigma range for Brightness channel adjustment is not valid.
        """

        # Check the intervals.
        #
        if hue_sigma_range is not None:
            if len(hue_sigma_range) != 2 or hue_sigma_range[1] < hue_sigma_range[0] or hue_sigma_range[0] < -1.0 or 1.0 < hue_sigma_range[1]:
                raise Exception("InvalidHueSigmaRangeError(hue_sigma_range)")

        if saturation_sigma_range is not None:
            if len(saturation_sigma_range) != 2 or saturation_sigma_range[1] < saturation_sigma_range[0] or saturation_sigma_range[0] < -1.0 or 1.0 < saturation_sigma_range[1]:
                raise Exception("InvalidSaturationSigmaRangeError(saturation_sigma_range)")

        if brightness_sigma_range is not None:
            if len(brightness_sigma_range) != 2 or brightness_sigma_range[1] < brightness_sigma_range[0] or brightness_sigma_range[0] < -1.0 or 1.0 < brightness_sigma_range[1]:
                raise Exception("InvalidBrightnessSigmaRangeError(brightness_sigma_range)")

        # Store the setting.
        #
        self.__sigma_ranges = [hue_sigma_range, saturation_sigma_range, brightness_sigma_range]

        self.__sigmas = [hue_sigma_range[0] if hue_sigma_range is not None else 0.0,
                         saturation_sigma_range[0] if saturation_sigma_range is not None else 0.0,
                         brightness_sigma_range[0] if brightness_sigma_range is not None else 0.0]

    def transform(self, patch):
        """
        Apply color deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Reorder the patch to channel last format and convert the image patch to HSB (=HSV) color coding.
        #
        patch_image = np.transpose(a=patch, axes=(1, 2, 0))
        patch_hsb = skimage.color.rgb2hsv(rgb=patch_image)

        # Augment the Hue channel.
        #
        if self.__sigmas[0] != 0.0:
            patch_hsb[:, :, 0] += self.__sigmas[0] % 1.0
            patch_hsb[:, :, 0] %= 1.0

        # Augment the Saturation channel.
        #
        if self.__sigmas[1] != 0.0:
            if self.__sigmas[1] < 0.0:
                patch_hsb[:, :, 1] *= (1.0 + self.__sigmas[1])
            else:
                patch_hsb[:, :, 1] *= (1.0 + (1.0 - patch_hsb[:, :, 1]) * self.__sigmas[1])

        # Augment the Brightness channel.
        #
        if self.__sigmas[2] != 0.0:
            if self.__sigmas[2] < 0.0:
                patch_hsb[:, :, 2] *= (1.0 + self.__sigmas[2])
            else:
                patch_hsb[:, :, 2] += (1.0 - patch_hsb[:, :, 2]) * self.__sigmas[2]

        # Convert back to RGB color coding with byte data type and order back to channels first order.
        #
        patch_rgb = skimage.color.hsv2rgb(hsv=patch_hsb)
        patch_rgb *= 255.0
        patch_rgb = patch_rgb.astype(dtype=np.uint8)
        patch_transformed = np.transpose(a=patch_rgb, axes=(2, 0, 1))

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma for each channel.
        #
        self.__sigmas = [np.random.uniform(low=sigma_range[0], high=sigma_range[1], size=None) if sigma_range is not None else 0.0 for sigma_range in self.__sigma_ranges]
