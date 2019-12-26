import numpy as np
import rasterio
from shapely.geometry import Point, Polygon
import rasterio.mask


class HighestPoint:

    def get_5kmbuffer(self, coordinate, background):
        point = Point(coordinate)
        bounds = background.bounds
        boundary = Polygon([(bounds.left, bounds.bottom), (bounds.right, bounds.bottom),
                            (bounds.right, bounds.top), (bounds.left, bounds.top)])
        buffer = point.buffer(5000).intersection(boundary)
        return buffer

    def row_to_xy(self, extend, row, col):
        x = extend[0] + col * extend[1] + row * extend[2]
        y = extend[3] + col * extend[4] + row * extend[5]
        point = (x, y)
        return point

    def find_hightest_point(self, extend, filepath, filename, buffer):
        file = filepath + "/" + filename
        shape = [buffer.__geo_interface__]
        with rasterio.open(file) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=False)
        m = np.argmax(out_image)
        row, col = divmod(m, out_image.shape[2])
        coordinate = self.row_to_xy(extend, row, col)
        return coordinate
