""" Available Transformations.

    Mainly wrappers around PIL classes.
"""
import numpy as np
from PIL import Image, ImageFilter


class Transformation(object):
    """ Interface for Transformation objects.
    """
    def __init__(self):
        self.info = NotImplemented

    def transform(self, img):
        raise NotImplementedError

    def get_info(self):
        if self.info is NotImplemented:
            raise NotImplementedError
        else:
            return self.info


class Scale(Transformation):
    def __init__(self, size, mutable=True):
        Transformation.__init__(self)

        if type(size) is tuple:
            self.w, self.h = size
            self.full_dims = True
            self.info = "_%dx%dpx" % (self.w, self.h)
        else:
            self.size = size
            self.full_dims = False
            self.info = "_%dpx" % (self.size)
        self.mutable = mutable

    def transform(self, img):
        if self.full_dims:
            return img.resize((self.w, self.h))
        else:
            return img.resize(self._get_new_dims(img))

    def _get_new_dims(self, img):
        w, h = img.size
        if h > w:
            return (int(self.size * h / w), self.size)
        elif w > h:
            return (self.size, int(self.size * h / w))
        else:
            return (self.size, self.size)


class Blur(Transformation):
    def __init__(self, radius, mutable=False):
        self.radius = radius
        self.mutable = mutable

        self.info = "_blur_%dpx" % (self.radius)
        self.gaussian_filter = ImageFilter.GaussianBlur(radius)

    def transform(self, img):
        return img.filter(self.gaussian_filter)


class ColorCast(Transformation):
    def __init__(self, angle, mutable=False):
        Transformation.__init__(self)

        self.h_step = 255 * angle / 360
        self.info = "_hue_%ddeg" % angle
        self.mutable = mutable

    def transform(self, img):
        return self._shift_hue(img)

    def _shift_hue(self, img, h_step=None):
        h_step = h_step or np.int(self.h_step)
        np_hsv = np.array(img.convert("HSV"))
        np_hsv[:, :, 0] = np.mod(np.add(np_hsv[:, :, 0], h_step), 255)
        return Image.fromarray(np_hsv, mode="HSV").convert("RGB")


class CycledColorCast(ColorCast):
    def __init__(self, angle, max_no=None):
        ColorCast.__init__(self, angle, False)

        self.max_no = max_no or int(360 / angle) - 1

    def transform(self, img):
        processed = []
        crt_h_step = 0
        for t in range(self.max_no):
            crt_h_step += self.h_step
            info = "_hue_%ddeg" % np.ceil(360 * crt_h_step / 255)
            processed.append((self._shift_hue(img, crt_h_step), info))
        return processed
