import json
from rtree import index


class NearestITN:
    def __init__(self):
        self.idx = index.Index()

    def get_itn(self, filepath, filename):
        itn_json = filepath + "/" + filename
        with open(itn_json, "r") as f:
            itn = json.load(f)
        return itn

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

    def nearest_node(self, test_point):
        for res in self.idx.nearest(test_point, 1, True):
            return res.object