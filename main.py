import numpy as np
from osgeo import gdal
import rasterio
from rasterio.windows import Window
from rasterio import plot
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import cartopy.crs as ccrs
import pyproj
import json
from rtree import index


class UserInput:

    def __init__(self, shape, x, y):  # takes two arguments: dem file, shapefile, xy inputs
        self.x = x
        self.y = y
        self.pt = Point(self.x, self.y)  # xy into shapely point
        self.shape = shape

    def within_mbr(self):
        mbr = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 8000)])  # create mbr shape

        while True:  # to test if point is within mbr and island shape
            if self.shape.contains(self.pt).all() & mbr.contains(self.pt):
                break  # loop stops if point is within mbr and within island
            else:
                print("Input coordinate outside box. Application quitting.")
                exit()  # app will quit if point is outside mbr and outside island


class HighestPoint:
    def get_mbr(self, coordinate):
        point = Point(coordinate)
        buffer = point.buffer(5000)
        mbr = buffer.bounds
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

    def find_hightest_point(self, extend, elevation, mbr):
        left_bottom = (mbr[0], mbr[1])
        top_right = (mbr[2], mbr[3])
        row_stop, col_start = self.xy_to_rowcol(extend, left_bottom)
        row_start, col_stop = self.xy_to_rowcol(extend, top_right)
        with elevation as src:
            w = src.read(1, window=Window.from_slices(slice(row_start, row_stop), slice(col_start, col_stop)))
        m = np.argmax(w)
        row, col = divmod(m, w.shape[1])
        return row, col

    def get_coordinate(self, extend, point, row, col):
        point_row, point_col = self.xy_to_rowcol(extend, point)
        coordinate_row = point_row + row - 1000
        coordinate_col = point_col + col - 1000
        coordinate = self.row_to_xy(extend, coordinate_row, coordinate_col)
        return coordinate


class NearestITN:
    def __init__(self):
        self.idx = index.Index()

    def nodes_index(self, itn):
        nodes = []
        node_names = []
        road_nodes = itn["roadnodes"]
        for name in road_nodes.keys():
            node_names.append(name)
        for node in road_nodes:
            road_node = road_nodes[node]
            coords = (road_node["coords"][0], road_node["coords"][1])
            nodes.append(coords)
        for n, node in enumerate(nodes):
            self.idx.insert(n, node, node_names[n])
        return self.idx

    def nearest_node(self, test_point, idx):
        for res in idx.nearest(test_point, 1, True):
            return res.object


class ShortestPath:
    def __init__(self):
        self.Graph = nx.Graph()

    def build_network(self, itn):
        road_links = itn['roadlinks']
        for link in road_links:
            self.Graph.add_edge(road_links[link]['start'], road_links[link]['end'], fid=link,
                                distance=road_links[link]['length'])
        network = self.Graph.to_directed()
        return network

    def xy_to_rowcol(self, extend, coordinate):
        a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
        b = np.array([coordinate[0] - extend[0], coordinate[1] - extend[3]])
        row_col = np.linalg.solve(a, b)
        row = int(np.floor(row_col[1]))
        col = int(np.floor(row_col[0]))
        return row, col

    def add_weight(self, extend, itn, network, height):
        road_nodes = itn['roadnodes']
        for (u, v, s) in network.edges(data='distance'):
            start_coord = road_nodes[u]["coords"]
            end_coord = road_nodes[v]["coords"]
            row, col = self.xy_to_rowcol(extend, start_coord)
            value = height[row, col]
            row1, col1 = self.xy_to_rowcol(extend, end_coord)
            value1 = height[row1, col1]
            if value1 > value:
                network[u][v]["time"] = (value1 - value) / 10 + (s * 3) / 250
            if value1 <= value:
                network[u][v]["time"] = (s * 3) / 250
        return network

    def shortest_path(self, start, end, network):
        shortest_path = nx.dijkstra_path(network, source=start, target=end, weight="time")
        return shortest_path


class ReadFile:
    def get_itn(self, filepath, filename):
        file = filepath + "/" + filename
        with open(file, "r") as f:
            itn = json.load(f)
        return itn

    def get_elevation(self, filepath, filename):
        file = filepath + "/" + filename
        dataset = gdal.Open(file)
        height = dataset.ReadAsArray()
        extend = dataset.GetGeoTransform()
        return extend, height

    def rasterio_read(self, filepath, filename):
        file = filepath + "/" + filename
        rasterio_read = rasterio.open(file)
        return rasterio_read

    def get_shape(self, filepath, filename):
        file = filepath + "/" + filename
        shape = gpd.read_file(file)
        return shape


def main():
    x = float(input("Enter location (as British National Grid coordinate)\nEasting: "))
    y = float(input("Northing: "))

    find_highest = HighestPoint()
    find_itn = NearestITN()
    find_path = ShortestPath()
    read_file = ReadFile()

    filepath_itn = "Material/Material/itn"
    filename_itn = "solent_itn.json"
    filepath_elevation = "Material/Material/elevation"
    filename_elevation = "SZ.asc"
    filepath_background = "Material/Material/background"
    filename_background = "raster-50k_2724246.tif"
    filepath_shape = "Material/Material/shape"
    filename_shape = "isle_of_wight.shp"

    itn = read_file.get_itn(filepath_itn, filename_itn)
    extend, height = read_file.get_elevation(filepath_elevation, filename_elevation)
    background = read_file.rasterio_read(filepath_background, filename_background)  # open background map
    elevation = read_file.rasterio_read(filepath_elevation, filename_elevation)  # open DEM map
    shape = read_file.get_shape(filepath_shape, filename_shape) # open shapefile

    shape_bng = shape.to_crs(epsg=27700)  # reproject shapefile from lat long to BNG
    shape_bng = shape_bng.set_geometry("geometry")  # set geomtry data
    test = UserInput(shape_bng, x, y)
    test.within_mbr()  # call function to test if point is within
    start_point = (x, y)

    start_point_mbr = find_highest.get_mbr(start_point)
    highest_row, highest_col = find_highest.find_hightest_point(extend, elevation, start_point_mbr)
    highest_point = find_highest.get_coordinate(extend, start_point, highest_row, highest_col)

    idx = find_itn.nodes_index(itn)
    start_node = find_itn.nearest_node(start_point, idx)
    end_node = find_itn.nearest_node(highest_point, idx)

    network_unweighted = find_path.build_network(itn)
    network_weighted = find_path.add_weight(extend, itn, network_unweighted, height)
    path = find_path.shortest_path(start_node, end_node, network_weighted)
