import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import Point, Polygon, LineString


class HighestPoint:

    # Get 5km buffer from user's position
    def get_5kmbuffer(self, coordinate, background):
        point = Point(coordinate)
        bounds = background.bounds
        boundary = Polygon([(bounds.left, bounds.bottom), (bounds.right, bounds.bottom),
                            (bounds.right, bounds.top), (bounds.left, bounds.top)])
        # Apply task 6, extend the region to whole island.
        # Get intersectional area as buffer to make sure programme do not crash out
        buffer = point.buffer(5000).intersection(boundary)
        return buffer

    # Define a function to transform row and column to projection coordinate
    def row_to_xy(self, extend, row, col):
        x = extend[0] + col * extend[1] + row * extend[2]
        y = extend[3] + col * extend[4] + row * extend[5]
        point = (x, y)
        return point

    # Define a function to find the highest point within 5km radius
    def find_hightest_point(self, extend, filepath, filename, buffer, start_point):
        file = filepath + "/" + filename
        shape = [buffer.__geo_interface__]
        # Use rasterio.mask function to clip array based on vector boundary
        # (Source: https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html)
        with rasterio.open(file) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=False)
            out_image = out_image.reshape(out_image.shape[1], out_image.shape[2])
        # Use np.where() and np.max() function to get index of maximum value in array
        a = np.where(out_image == np.max(out_image))
        # Determine the nearest highest point if the number of highest point is more than 1
        # Calculate Euclidean distance
        length_a = len(a[0])
        if length_a == 1:
            row, col = a[0][0], a[1][0]
        if length_a > 1:
            for i in range(length_a):
                row_x = a[0][i]
                col_y = a[1][i]
                x = self.row_to_xy(extend, row_x, col_y)[0]
                y = self.row_to_xy(extend, row_x, col_y)[1]
                length = (x - float(start_point[0])) ** 2 + (y - float(start_point[1])) ** 2
                distance = np.sqrt(length)
                if i == 0:
                    shortest = distance
                    row = row_x
                    col = col_y
                else:
                    if distance < shortest:
                        shortest = distance
                        row = row_x
                        col = col_y
        coordinate = self.row_to_xy(extend, row, col)
        return coordinate

