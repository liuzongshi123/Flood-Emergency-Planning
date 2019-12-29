import numpy as np
from osgeo import gdal
import rasterio
import rasterio.mask
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
import networkx as nx
import geopandas as gpd
import cartopy.crs as ccrs
import json
from rtree import index
from collections import OrderedDict
import os
import argparse
from tkinter import *


class UserInput:

    # Set GUI for user input their location
    def __init__(self, master):
        self.frame = master
        self.lab = Label(self.frame, text='Please enter your location \n(as British National Grid coordinate)',
                         font=('Arial', 14))
        self.lab.pack()
        self.easting = Label(self.frame, text="Easting:", font=('Arial', 12))
        self.easting.pack()
        self.enter_east = Entry(self.frame)
        self.enter_east.pack()
        self.northing = Label(self.frame, text="Northing:", font=('Arial', 12))
        self.northing.pack()
        self.enter_north = Entry(self.frame)
        self.enter_north.pack()
        self.button_sub = Button(self.frame, text="Submit", command=self.submit)
        self.button_sub.pack()
        self.lab_remind = Label(self.frame, text="")
        self.lab_remind.pack()
        self.button_quit = Button(self.frame, text="Quit", command=self.quit)
        self.button_quit.pack()
        self.x = None
        self.y = None

    # To test if point is within island shape
    # Programme will ask you to enter location again if entered point is outside island
    def submit(self):
        x = self.enter_east.get()
        y = self.enter_north.get()
        pt = Point(float(x), float(y))
        shape = gpd.read_file(os.getcwd() + "/" + "Material/shape/isle_of_wight.shp")
        shape_bng = shape.to_crs(epsg=27700)  # reproject shapefile from lat long to BNG
        shape_bng = shape_bng.set_geometry("geometry")  # set geomtry data
        if not shape_bng.contains(pt).all():
            if not shape_bng.touches(pt).all():
                self.lab_remind["text"] = "Input coordinate outside Isle of Wight.\nPlease Try Again!"
                self.enter_east.delete(0, len(x))
                self.enter_north.delete(0, len(y))
        else:
            self.x = pt.x
            self.y = pt.y
            self.frame.destroy()

    def quit(self):
        exit()


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
        return coordinate, out_image, row, col

    # Define a function to refind highest point if there are some parts of shortest path outside the 5km radius
    def refind_hightest_point(self, extend, out_image, row_highest, col_highest):
        out_image[row_highest, col_highest] = 0
        m = np.argmax(out_image)
        row, col = divmod(m, out_image.shape[1])
        coordinate = self.row_to_xy(extend, row, col)
        return coordinate, out_image, row, col


class NearestITN:

    def __init__(self):
        self.idx = index.Index()

    # Insert rode nodes into r-tree index
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

    # Use r-tree index to search the nearest road node
    def nearest_node(self, test_point, idx):
        for res in idx.nearest(test_point, 1, True):
            return res.object


class ShortestPath:

    def __init__(self):
        self.Graph = nx.Graph()

    # Use networkX to build digraph
    def build_network(self, itn):
        road_links = itn['roadlinks']
        for link in road_links:
            self.Graph.add_edge(road_links[link]['start'], road_links[link]['end'], fid=link,
                                distance=road_links[link]['length'])
        network = self.Graph.to_directed()
        return network

    # Define a function to transform projection coordinate to row and column in array
    def xy_to_rowcol(self, extend, coordinate):
        a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
        b = np.array([coordinate[0] - extend[0], coordinate[1] - extend[3]])
        # Use np.linalg.solve() to get row and column location in array
        # (Source: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html)
        row_col = np.linalg.solve(a, b)
        row = int(np.floor(row_col[1]))
        col = int(np.floor(row_col[0]))
        return row, col

    # Add "time" weight to each edge in digraph by applying Naismith’s rule
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

    # Get shortest path based on "time" weight
    def shortest_path(self, start, end, network):
        shortest_path = nx.dijkstra_path(network, source=start, target=end, weight="time")
        return shortest_path


class ReadFile:
    # Develop four functions to read file from filepath
    def get_itn(self, filepath, filename):
        file = filepath + "/" + filename
        with open(file, "r") as f:
            itn = json.load(f)
        return itn

    def get_extend(self, filepath, filename):
        file = filepath + "/" + filename
        dataset = gdal.Open(file)
        extend = dataset.GetGeoTransform()
        return extend

    def rasterio_read(self, filepath, filename):
        file = filepath + "/" + filename
        rasterio_read = rasterio.open(file)
        return rasterio_read


class Plotter:

    # Define a function to transform projection coordinate to row and column in array
    def xy_to_rowcol(self, extend, coordinate):
        a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
        b = np.array([coordinate[0] - extend[0], coordinate[1] - extend[3]])
        row_col = np.linalg.solve(a, b)
        row = int(np.floor(row_col[1]))
        col = int(np.floor(row_col[0]))
        return row, col

    # Plot 1:50k Ordnance Survey background map of the surrounding area
    def add_background(self, ax, extend, background, bounds, extent):
        row_stop, col_start = self.xy_to_rowcol(extend, (bounds[0], bounds[1]))
        row_start, col_stop = self.xy_to_rowcol(extend, (bounds[2], bounds[3]))
        # Clip the raster to surrounding area by using rasterio.window function
        back_array = background.read(1, window=Window.from_slices(slice(row_start, row_stop),
                                                                  slice(col_start, col_stop)))
        palette = np.array([value for key, value in background.colormap(1).items()])
        background_image = palette[back_array]
        ax.imshow(background_image, origin="upper", extent=extent, zorder=1)

    # Overlay a transparent elevation raster within 5km radius
    def add_elevation(self, ax, buffer, elevation, extent):
        # Clip elevation array into minimum bounding rectangle of 5km radius
        feature = [buffer.__geo_interface__]
        out_image, out_transform = rasterio.mask.mask(elevation, feature, crop=True)
        out_image = out_image.reshape(out_image.shape[1], out_image.shape[2])
        # Clip the elevation image based on boundary of 5km circle
        x, y = buffer.exterior.xy
        array = []
        for i in range(len(x) - 1):
            array.append([x[i], y[i]])
        elevation_array = np.array(array)
        patch = patches.Polygon(elevation_array, closed=True, transform=ax.transData)
        ax.imshow(out_image, origin="upper", extent=extent, cmap="terrain", zorder=2, alpha=0.5,
                  clip_path=patch)

    # Plot shorestest path
    def add_shortest_path(self, ax, itn, path, network_weighted):
        road_links = itn["roadlinks"]
        links = []  # this list will be used to populate the feature id (fid) column
        geom = []  # this list will be used to populate the geometry column
        first_node = path[0]
        for node in path[1:]:
            link_fid = network_weighted.edges[first_node, node]['fid']
            links.append(link_fid)
            geom.append(LineString(road_links[link_fid]['coords']))
            first_node = node
        # create a GeoDataFrame of the shortest path
        shortest_path_gpd = gpd.GeoDataFrame({"fid": links, "geometry": geom})
        shortest_path_gpd.plot(ax=ax, edgecolor="blue", linewidth=1, zorder=3, label='Shortest Path')

    # Plot start road node and destination
    def add_points(self, itn, start_node, end_node):
        road_nodes = itn["roadnodes"]
        plt.plot(road_nodes[start_node]["coords"][0], road_nodes[start_node]["coords"][1], "ro", label='Start Point',
                 markersize=3, zorder=4)
        plt.plot(road_nodes[end_node]["coords"][0], road_nodes[end_node]["coords"][1], "go", label='Highest Point',
                 markersize=3, zorder=4)

    # Plot color bar showing elevation range
    def add_colorbar(self, ax, fig, buffer, elevation):
        feature = [buffer.__geo_interface__]
        out_image, out_transform = rasterio.mask.mask(elevation, feature, crop=True)
        out_image = out_image.reshape(out_image.shape[1], out_image.shape[2])
        cax = ax.contourf(out_image, np.arange(0, 251, 10), cmap="terrain", origin='upper', zorder=0)
        fig.colorbar(cax, ax=ax)

    # Add sacle bar by using matplotlib_scalebar library
    def add_scalebar(self, ax):
        scalebar = ScaleBar(5, location="lower left")
        ax.add_artist(scalebar)

    # Draw a north arrow
    # (explanation of param:
    # Param loc_x: the horizontal proportion of the entire ax centered at the bottom of the text,
    # Param loc_y: the proportion of the entire ax axis with the bottom of the text as the center,
    # Param width: the compass's proportion of ax width,
    # Param height: the proportion of the compass to the height of ax,
    # Param pad: text symbols account for ax proportion gap)
    # Source: (https://blog.csdn.net/weixin_44092702/article/details/99690821)
    def add_north(self, ax, loc_x=0.07, loc_y=0.95, width=0.07, height=0.1, pad=0.12):
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ylen = maxy - miny
        xlen = maxx - minx
        left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
        right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
        top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
        center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
        array = np.array([left, top, right, center])
        triangle = patches.Polygon(array, closed=True, facecolor='k', transform=ax.transData, zorder=7)
        ax.add_patch(triangle)

    # Add text "N" for north arrow
    def add_text(self, ax, labelsize=8, loc_x=0.07, loc_y=0.92):
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ax.text(s='N',
                x=minx + maxx * loc_x,
                y=miny + maxy * loc_y,
                fontsize=labelsize,
                ha='center',
                va='bottom', transform=ax.transAxes, color="k", zorder=8)

    # Show labels and figure
    def show(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 5})
        plt.show()


# Get the location from user and plot the shortest path
# Pass file path as arguments to function.
def main(filepath):
    # Set GUI for user input their location
    root = Tk()
    root.title("Flood Emergency Planning")
    root.geometry('400x300')
    pt = UserInput(root)
    root.mainloop()
    start_point = (pt.x, pt.y)
    print("Programme is running, please wait a moment!")

    find_highest = HighestPoint()
    find_itn = NearestITN()
    find_path = ShortestPath()
    read_file = ReadFile()
    plotter = Plotter()

    filepath_itn = filepath + "/" + "Material/itn"
    filename_itn = "solent_itn.json"
    filepath_elevation = filepath + "/" + "Material/elevation"
    filename_elevation = "SZ.asc"
    filepath_background = filepath + "/" + "Material/background"
    filename_background = "raster-50k_2724246.tif"

    itn = read_file.get_itn(filepath_itn, filename_itn)
    extend = read_file.get_extend(filepath_elevation, filename_elevation)
    extend_back = read_file.get_extend(filepath_background, filename_background)
    background = read_file.rasterio_read(filepath_background, filename_background)
    elevation = read_file.rasterio_read(filepath_elevation, filename_elevation)

    start_point_buffer = find_highest.get_5kmbuffer(start_point, background)
    highest_point, out_image, row_highest, col_highest = \
        find_highest.find_hightest_point(
            extend, filepath_elevation, filename_elevation, start_point_buffer, start_point)

    idx = find_itn.nodes_index(itn)
    start_node = find_itn.nearest_node(start_point, idx)
    end_node = find_itn.nearest_node(highest_point, idx)

    network_unweighted = find_path.build_network(itn)
    network_weighted = find_path.add_weight(extend, itn, network_unweighted, elevation.read(1))
    shortest_path = find_path.shortest_path(start_node, end_node, network_weighted)

    # Define a function to solve some special cases in northern island
    # Determine if the shortest path cross the boundary of 5km radius
    # If shortest path cross the boundary, delete this point, then find highest point again
    road_links = itn["roadlinks"]
    road = []
    first_node = shortest_path[0]
    for node in shortest_path[1:]:
        link_fid = network_weighted.edges[first_node, node]['fid']
        for i in road_links[link_fid]['coords']:
            road.append(i)
        first_node = node
    path_line = LineString(road)
    if path_line.crosses(start_point_buffer):
        print("You are in the special area of island, it will take a long time for programme running!"
              "Sorry about that!")
    while path_line.crosses(start_point_buffer):
        highest_point, out_image, row_highest, col_highest = \
            find_highest.refind_hightest_point(extend, out_image, row_highest, col_highest)
        end_node = find_itn.nearest_node(highest_point, idx)
        shortest_path = find_path.shortest_path(start_node, end_node, network_weighted)
        road = []
        first_node = shortest_path[0]
        for node in shortest_path[1:]:
            link_fid = network_weighted.edges[first_node, node]['fid']
            for i in road_links[link_fid]['coords']:
                road.append(i)
            first_node = node
        path_line = LineString(road)

    bounds = start_point_buffer.bounds
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.OSGB())

    plotter.add_text(ax)
    plotter.add_background(ax, extend_back, background, bounds, extent)
    plotter.add_elevation(ax, start_point_buffer, elevation, extent)
    plotter.add_shortest_path(ax, itn, shortest_path, network_weighted)
    plotter.add_points(itn, start_node, end_node)
    plotter.add_north(ax)
    plotter.add_colorbar(ax, fig, start_point_buffer, elevation)
    plotter.add_scalebar(ax)
    ax.set_extent(extent, crs=ccrs.OSGB())
    plotter.show()


if __name__ == '__main__':
    # Use "os" library to get current work directory
    current_work_directory = os.getcwd()
    # Use "argparse" library to get filepath argument
    parser = argparse.ArgumentParser(description="Welcome to Flood Emergency Planning\n"
                                                 "Please enter file path as arguments")
    # Get "filepath" argument, default value is current work directory.
    parser.add_argument("-f", "--filepath",
                        help="filepath, optional arguments, default is current work directory",
                        default=current_work_directory)
    args = parser.parse_args()
    # Run program, print wrong message when program do not work
    try:
        main(args.filepath)
    except Exception as e:
        print(e)
    # Programme will not shut down immediately if it do not work correctly
    os.system("pause")
