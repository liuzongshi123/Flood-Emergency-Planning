import json
import networkx as nx
import numpy as np
from osgeo import gdal


class ShortestPath:
    def __init__(self):
        self.Graph = nx.Graph()

    def get_itn(self, filepath, filename):
        itn_json = filepath + "/" + filename
        with open(itn_json, "r") as f:
            itn = json.load(f)
        return itn

    def get_elevation(self, filepath, filename):
        file = filepath + "/" + filename
        dataset = gdal.Open(file)
        elevation = dataset.ReadAsArray()
        extend = dataset.GetGeoTransform()
        return extend, elevation

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

    def add_weight(self, itn, network, elevation):
        road_nodes = itn['roadnodes']
        for (u, v, s) in network.edges(data='distance'):
            start_coord = road_nodes[u]["coords"]
            end_coord = road_nodes[v]["coords"]
            row, col = self.xy_to_rowcol(extend, start_coord)
            value = elevation[row, col]
            row1, col1 = self.xy_to_rowcol(extend, end_coord)
            value1 = elevation[row1, col1]
            if value1 > value:
                network[u][v]["time"] = (value1 - value) / 10 + (s * 3) / 250
            if value1 <= value:
                network[u][v]["time"] = (s * 3) / 250
        return network

    def shortest_path(self, start, end, network):
        shortest_path = nx.dijkstra_path(network, source=start, target=end, weight="time")
        return shortest_path

















