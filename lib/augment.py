import math
import os
import random
import tempfile
import torch
import uuid
from PIL import Image
from PIL import ImageEnhance

# From https://stackoverflow.com/a/16778797/10444046
def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return int(wr), int(hr)


class AugmentTransforms():
    def __init__(self, config):
        self.config = config

    def rotate_with_crop(self, image: Image, angle):
        x, y = image.size
        X, Y = rotatedRectWithMaxArea(x, y, math.radians(random.randint(-angle, angle)))
        dx, dy = (x - X) // 2, (y - Y) // 2
        return image.rotate(angle, expand=False, fillcolor=(255,255,255), resample=3).crop(
            (dx, dy, dx + X, dy + Y)
        )

    def flip(self, image, x):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def adjust_contrast(self, image, factor):
        enh_con = ImageEnhance.Contrast(image)
        return enh_con.enhance(factor)

    def adjust_brightness(self, image, brightness):
        enh_bri = ImageEnhance.Brightness(image)
        return enh_bri.enhance(brightness)

    def adjust_color(self, image, color):
        enh_col = ImageEnhance.Color(image)
        return enh_col.enhance(color)

    def transform(self, img, roll):
        if not self.config.enabled or roll < self.config.dropout:
            return img

        raw_img = img
        for method, factor in self.config.methods.items():
            img = getattr(self, method)(img, factor)

        if self.config.debug:
            basedir = os.path.join(tempfile.gettempdir(), "nd-aug-debug")
            os.makedirs(basedir, exist_ok=True)
            filename = "aug_" + str(uuid.uuid4())[:8]
            rawp = os.path.join(basedir, f"{filename}_raw.jpg")
            trsp = os.path.join(basedir, f"{filename}_transformed.jpg")

            raw_img.save(rawp)
            img.save(trsp)
            print(f"saved: {rawp}")
            print(f"saved: {trsp}")

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
