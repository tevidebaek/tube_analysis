'''
This codes provides a GUI that one can use to analyze tubes.

What do we want this to do? We want to iterate through a series of images and 
extract the tube width, angle, and lattice angle from the image. After getting 
these we will save this information as well as relevant plots from the lattice
process. After saving, go to the next image.
'''

from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import tube_analysis as tube
import pandas as pd
import os
import pims

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)
        
        self.tube_param = []
        self.lattice_param = []

        self.counter = 0

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Save", command=self.save)
        file.add_command(label="Exit", command=self.client_exit)
        
        menu.add_cascade(label="File", menu=file)
        analyze = Menu(menu)

        analyze.add_command(label="Region of Interest", 
        command=self.regionOfInterest)
        
        analyze.add_command(label="Get Tube Parameters", 
        command=self.widthAndAngle)
        
        analyze.add_command(label="Lattice Angle - FFT", 
        command=self.latticeFFT)
        
        LFFT = Button(self.master, text='lattice angle', command=self.latticeFFT)

        menu.add_cascade(label="Analyze", menu=analyze)
        load = Image.open(image_src)
        render = ImageTk.PhotoImage(load)

        self.image = load

        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

    def regionOfInterest(self):
        root.config(cursor="plus")
        canvas.bind("<Button-1>", self.imgClick)
        
    def widthAndAngle(self):
        #print(self.pos)
        
        #the first two points define the side of the tube
        angle = np.arctan2(self.pos[0][1]-self.pos[1][1], self.pos[1][0]-self.pos[0][0]) #the first y points are inverted due to the image being displayed with y=0 at the top
        if angle>90: angle -= 180
        if angle<-90: angle +=180
        
        print('tube angle: ', angle*180./np.pi)

        #m = (self.pos[1][1]-self.pos[0][1])/(self.pos[1][0]-self.pos[0][0])
        
        def side_length( pt1, pt2):
            return np.sqrt((pt1[1]-pt2[1])**2 + (pt1[0]-pt2[0])**2)
        
        b = side_length(self.pos[0], self.pos[1])
        a = side_length(self.pos[1], self.pos[2])
        c = side_length(self.pos[2], self.pos[0])
        
        #use Heron's formula
        h = np.sqrt( (a+b+c)*(-a+b+c)*(a-b+c)*(a+b-c) )/(2*b)
        print('tube width: ', h)
        
        self.tube_param = [h, angle*180./np.pi]

    def latticeFFT(self):
        '''
        The following section finds the orientation of the triangular
        lattice on the tube.
        '''
        
        fft = tube.get_fft(self.image)
        
        orients = np.linspace(0,59.5, num=60*2)
        a_s = np.linspace(10.0, 30.0, num = 20*4+1)
        
        da, do = a_s[1]-a_s[0], orients[1]-orients[0]
        
        ints = []
    
        for a in a_s:
            ints = tube.analyze_a_kernels(a,fft,ints)
            
        ints = np.array(ints)
        shape_a_o = (len(a_s),len(orients))
        
        a_peak, a_width, o_peak, o_width = tube.fwhm_width(ints, shape_a_o, da, do)
        print('fit parameters from FWHM:')
        print('lattice spacing: ', a_peak)
        print('lattice spacing error', a_width)
        print('lattice angle: ', o_peak)
        print('lattice angle error', o_width)
        
        self.lattice_param = [a_peak, a_width, o_peak, o_width]

    def save(self):
        '''
        saves the lattice and tube parameters into a CSV file
        after saving, clears the position and parameter values
        '''
        
        spc_l, spc_e, ang_l, ang_e = self.lattice_param
        tube_w, tube_ang = self.tube_param
    
        #check if there is a file to open
        if os.path.exists(src_save+filename_save): df = pd.read_pickle(src_save+filename_save)
        else : df = pd.DataFrame()
        
        df = df.append({'a':spc_l, 'da':spc_e, 'lat_ang':ang_l, 'lat_err':ang_e, 'width':tube_w, 'ang':tube_ang},ignore_index=True)
        df.to_pickle(src_save+filename_save)

        #clear the stored data in the system
        self.pos = []
        self.lattice_param = []
        self.tube_param = []
        print('saved')

    def client_exit(self):
        self.master.destroy()

    def imgClick(self, event):

        if self.counter < 3:
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            self.pos.append((x, y))
            #print(self.pos)
            canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair")
            canvas.create_line(x, y - 5, x, y + 5, fill="red", tags="crosshair")
            self.counter += 1
        else:
            canvas.unbind("<Button 1>")
            root.config(cursor="arrow")
            self.counter = 0
        

image_dir = 'C:/Users/tevid/Desktop/DNA_Origami/Code/TubeAnalysis/images/'

#location of the .pkl file that we want to save image info to
src_save = 'C:/Users/tevid/Desktop/DNA_Origami/Code/TubeAnalysis/data/'
filename_save = 'test_file.pkl'

for dirpath, dirnames, filenames in os.walk(image_dir):
    for filename in filenames:
        if filename.endswith('.png'):
            
            print(filename)

            image_src = image_dir+filename
            
            root = Tk()
            imgSize = Image.open(image_src)
            tkimage =  ImageTk.PhotoImage(imgSize)
            w, h = imgSize.size
    
            canvas = Canvas(root, width=w, height=h)
            canvas.create_image((w/2,h/2),image=tkimage)
            canvas.pack()
    
            root.geometry("%dx%d"%(w,h))
            app = Window(root)
            root.mainloop()
    
            print('this should print after mainloop is ended')