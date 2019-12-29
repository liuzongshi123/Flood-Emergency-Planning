from shapely.geometry import Point
import geopandas as gpd
import os
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