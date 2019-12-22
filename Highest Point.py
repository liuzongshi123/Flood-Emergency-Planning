import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point


class HighestPoint:
    def get_mbr(self, coordinate):
        point = Point(coordinate)
        buffer = point.buffer(5000)
        mbr = buffer. bounds
        return mbr

    def xy_to_rowcol(self, extend, coordinate):
        a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
        b = np.array([coordinate[0] - extend[0], coordinate[1] - extend[3]])
        row_col = np.linalg.solve(a, b)
        row = int(np.floor(row_col[1]))
        col = int(np.floor(row_col[0]))
        return row, col

    def row_to_xy(self, extend, row, col):
        x = extend[0] + col * extend[1] + row * extend[2]
        y = extend[3] + col * extend[4] + row * extend[5]
        point = (x, y)
        return point

    def find_hightest_point(self, filepath, filename, mbr):
        file = filepath + "/" + filename
        left_bottom = (mbr[0], mbr[1])
        top_right = (mbr[2], mbr[3])
        row_stop, col_start = self.xy_to_rowcol(extend, left_bottom)
        row_start, col_stop = self.xy_to_rowcol(extend, top_right)
        with rasterio.open(file) as src:
            w = src.read(1, window=Window.from_slices(slice(row_start, row_stop), slice(col_start, col_stop)))
        m = np.argmax(w)
        row, col = divmod(m, w.shape[1])
        return row, col

    def get_coordinate(self, point, row, col):
        point_row, point_col = self.xy_to_rowcol(extend, point)
        coordinate_row = point_row + row - 1000
        coordinate_col = point_col + col - 1000
        coordinate = self.row_to_xy(extend, coordinate_row, coordinate_col)
        return coordinate

