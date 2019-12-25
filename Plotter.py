import rasterio
import rasterio.mask
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, LineString
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import geopandas as gpd


class Plotter:

    def xy_to_rowcol(self, extend, coordinate):
        a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
        b = np.array([coordinate[0] - extend[0], coordinate[1] - extend[3]])
        row_col = np.linalg.solve(a, b)
        row = int(np.floor(row_col[1]))
        col = int(np.floor(row_col[0]))
        return row, col

    def add_background(self, ax, extend, background, bounds, extent):
        row_stop, col_start = self.xy_to_rowcol(extend, (bounds[0], bounds[1]))
        row_start, col_stop = self.xy_to_rowcol(extend, (bounds[2], bounds[3]))
        back_array = background.read(1, window=Window.from_slices(slice(row_start, row_stop),
                                                                  slice(col_start, col_stop)))
        palette = np.array([value for key, value in background.colormap(1).items()])
        background_image = palette[back_array]
        ax.imshow(background_image, origin="upper", extent=extent, zorder=1)

    def add_elevation(self, ax, buffer, elevation, extent):
        feature = [buffer.__geo_interface__]
        out_image, out_transform = rasterio.mask.mask(elevation, feature, crop=True)
        out_image = out_image.reshape(out_image.shape[1], out_image.shape[2])
        x, y = buffer.exterior.xy
        array = []
        for i in range(len(x) - 1):
            array.append([x[i], y[i]])
        elevation_array = np.array(array)
        patch = patches.Polygon(elevation_array, closed=True, transform=ax.transData)
        ax.imshow(out_image, origin="upper", extent=extent, cmap="terrain", zorder=2, alpha=0.5,
                  clip_path=patch)

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
        shortest_path_gpd = gpd.GeoDataFrame({"fid": links, "geometry": geom})
        shortest_path_gpd.plot(ax=ax, edgecolor="blue", linewidth=1, zorder=3, label='Shortest Path')

    def add_points(self, itn, start_node, end_node):
        road_nodes = itn["roadnodes"]
        plt.plot(road_nodes[start_node]["coords"][0], road_nodes[start_node]["coords"][1], "ro", label='Start Point',
                 markersize=3, zorder=4)
        plt.plot(road_nodes[end_node]["coords"][0], road_nodes[end_node]["coords"][1], "go", label='Highest Point',
                 markersize=3, zorder=4)

    def add_colorbar(self, ax, fig, buffer, elevation):
        feature = [buffer.__geo_interface__]
        out_image, out_transform = rasterio.mask.mask(elevation, feature, crop=True)
        out_image = out_image.reshape(out_image.shape[1], out_image.shape[2])
        cax = ax.contourf(out_image, np.arange(0, 251, 10), cmap="terrain", origin='upper', zorder=0)
        fig.colorbar(cax, ax=ax)

    def add_scalebar(self, ax):
        scalebar = ScaleBar(5, location="lower left")
        ax.add_artist(scalebar)

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

    def add_text(self, ax, labelsize=8, loc_x=0.07, loc_y=0.92):
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ax.text(s='N',
                x=minx + maxx * loc_x,
                y=miny + maxy * loc_y,
                fontsize=labelsize,
                ha='center',
                va='bottom', transform=ax.transAxes, color="k", zorder=8)

    def show(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 5})
        plt.show()


