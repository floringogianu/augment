""" Available Transformations.

    Mainly wrappers around PIL classes.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageTransform
from termcolor import colored as clr


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
            return (int(self.size * w / h), self.size)
        elif w > h:
            return (self.size, int(self.size * h / w))
        else:
            return (self.size, self.size)


class RadialDistortion(object):
    def __init__(self):
        pass

    def _normalize(self, dst_grid, w, h):
        dst_grid[:, :, 0] = np.divide(np.multiply(dst_grid[:, :, 0], 2) - w, w)
        dst_grid[:, :, 1] = np.divide(np.multiply(dst_grid[:, :, 1], 2) - h, h)
        return dst_grid

    def _denormalize(self, dst_grid, w, h):
        dst_grid[:, :, 0] = ((dst_grid[:, :, 0] + 1) * w) / 2
        dst_grid[:, :, 1] = ((dst_grid[:, :, 1] + 1) * h) / 2
        return dst_grid

    def _get_l2(self, dst_grid):
        return np.add(np.power(dst_grid[:, :, 0], 2),
                      np.power(dst_grid[:, :, 1], 2))


class PincushionDistortion(RadialDistortion):
    def __init__(self, k):
        RadialDistortion.__init__(self)
        if k < 0:
            raise ValueError(clr(
                "Distortion coefficient can't be negative for" +
                " pincushion effects. Try T.BarrelDistortion.",
                "white", "on_red"))
        self.k = k
        self.info = "_pincushion_dst"

    def distort_grid(self, src_grid, w, h):
        dst_grid = src_grid.copy().astype(np.float)
        dst_grid = self._normalize(dst_grid, w, h)

        r = self._get_l2(dst_grid)

        # Apply the radial distortion model.
        dst_grid[:, :, 0] = dst_grid[:, :, 0] * (1 - self.k * r)
        dst_grid[:, :, 1] = dst_grid[:, :, 1] * (1 - self.k * r)

        dst_grid = self._denormalize(dst_grid, w, h)

        return dst_grid.astype(np.int16)


class BarrelDistortion(PincushionDistortion):
    def __init__(self, k):
        PincushionDistortion.__init__(self, k)
        print("Warning, BarrelDistortion requires cropping, " +
              "will be implemented soon.")
        if k < 0:
            raise ValueError(clr(
                "Distortion coefficient shoule be positive.",
                "white", "on_red"))
        self.k = -k
        self.info = "_barrel_dst"


class RandomDistortion(object):
    def __init__(self, kx, ky):
        RadialDistortion.__init__(self)
        self.kx = kx
        self.ky = ky


class Distort(Transformation):
    def __init__(self, kind=BarrelDistortion(k=0.125), grid_div=(4, 4),
                 mutable=False):
        Transformation.__init__(self)

        self.grid_div = grid_div
        self.kind = kind
        self.mutable = mutable
        self.info = self.kind.info

    def transform(self, img):
        src_grid = self._compute_grid(*img.size, *self.grid_div)
        dst_grid = self.kind.distort_grid(src_grid, *img.size)
        mesh = self._grid_to_mesh(dst_grid, src_grid)
        return img.transform(img.size, ImageTransform.MeshTransform(mesh))

    def _compute_grid(self, w, h, w_div, h_div):
        """ Compute grid coordinates and generate the meshgrid. """
        x_grid = np.arange(0, w + 1, w / w_div)
        y_grid = np.arange(0, h + 1, h / h_div)
        mesh_grid = np.meshgrid(x_grid, y_grid)
        return np.asarray(mesh_grid,
                          dtype=np.int16).swapaxes(0, 2).swapaxes(0, 1)

    def _grid_to_mesh(self, dst_grid, src_grid):
        assert(src_grid.shape == dst_grid.shape)
        mesh = []
        for i in range(dst_grid.shape[0] - 1):
            for j in range(dst_grid.shape[1] - 1):

                dst_quad = self._map_to_quad(dst_grid, i, j)
                src_quad = self._map_to_quad(src_grid, i, j)

                src_rect = self._quad_to_rect(src_quad)
                mesh.append([src_rect, dst_quad])
        return mesh

    def _map_to_quad(self, grid, i, j):
        return [grid[i, j, 0], grid[i, j, 1],
                grid[i + 1, j, 0], grid[i + 1, j, 1],
                grid[i + 1, j + 1, 0], grid[i + 1, j + 1, 1],
                grid[i, j + 1, 0], grid[i, j + 1, 1]]

    def _quad_as_rect(self, quad):
        if quad[0] != quad[2]:
            return False
        elif quad[1] != quad[7]:
            return False
        elif quad[4] != quad[6]:
            return False
        elif quad[3] != quad[5]:
            return False
        else:
            return True

    def _quad_to_rect(self, quad):
        assert(len(quad) == 8)
        assert(self._quad_as_rect(quad))
        return (quad[0], quad[1], quad[4], quad[3])


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
