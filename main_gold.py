import rasterio
from rasterio import plot
import pyproj
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


def main():
    background = rasterio.open("background/raster-50k_2724246.tif")  # open background map
    elevation = rasterio.open("elevation/SZ.asc")  # open DEM map
    shape = gpd.read_file("shape/isle_of_wight.shp")  # open shapefile

    shape_bng = shape.to_crs(epsg=27700)  # reproject shapefile from lat long to BNG
    shape_bng = shape_bng.set_geometry("geometry")  # set geomtry data

    x = float(input("Enter location (as British National Grid coordinate)\nEasting: "))
    y = float(input("Northing: "))

    test = UserInput(elevation, shape_bng, x, y)
    test.within_mbr()  # call function to test if point is within

    mapping = MapPlotting()  # mapping
    mapping.add_maps(background, elevation)
    mapping.add_point(x, y)
    mapping.show()


class UserInput:

    def __init__(self, elevation, shape, x, y):  # takes two arguments: dem file, shapefile, xy inputs
        self.x = x
        self.y = y
        self.pt = Point(self.x, self.y)  # xy into shapely point
        self.elevation = elevation
        self.shape = shape

    def within_mbr(self):
        mbr = Polygon([(430000, 80000), (430000, 95000), (465000, 95000), (465000, 8000)])  # create mbr shape

        while True:  # to test if point is within mbr and island shape
            if self.shape.contains(self.pt).all() & mbr.contains(self.pt):
                break  # loop stops if point is within mbr and within island
            else:
                print("Input coordinate outside box. Application quitting.")
                exit()  # app will quit if point is outside mbr and outside island


# class HighestPoint(UserInput):


# class NearestITN(HighestPoint):


# class ShortestPath(NearestITN):


class MapPlotting:

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 15))

    def add_maps(self, background, elevation):  # add base and elevation maps
        rasterio.plot.show(background, ax=self.ax)  # COLOUR MAP NEEDS TO BE CHANGED
        rasterio.plot.show(elevation, ax=self.ax, alpha=0.5)  # change DEM overlay opacity

    def add_point(self, x, y):  # add user starting point
        plt.plot(x, y, "ro")

    # def add_path(self):  # add shortest path

    def show(self):
        plt.show()


if __name__ == "__main__":
    main()
