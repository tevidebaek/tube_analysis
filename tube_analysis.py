# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:27:26 2020

@author: tevid
"""

#this is a collection of functions used in the image analysis
#for tube structures. It has two primary tasks:
# (i) identify the orientation of the triangular lattice
#(ii) find the orientation and width of the tubes

import numpy as np
import pandas as pd
import pims
import scipy.ndimage as ndim
from scipy import optimize
import skimage.filters
from scipy.ndimage import measurements
import matplotlib.pyplot as plt

##################################

def make_triang_kernel(spacing,orientation,shape):
    '''
    This will construct a kernel that we will use to integrate over the FFT.
    If the lattice spacing and orientation match, it should give a maximum signal.
    The inputs are the lattice spacing, the lattice angle, and the FFT image shape.

    Parameters
    ----------
    spacing : float
        the lattice spacing for the triangular points
    orientation : float
        angle orientation of the kernel lattice (in degrees)
    shape : tuple
        a tuple of (N_rows, M_columns) for kernel shape

    Returns
    -------
    kernel : ndarray
        ndarray of dimension (N,M) with Gaussian peaks of amplitude 1/(num peaks)
    points : ndarray
        ndarray of shape (2,num peaks) with x,y positions of centers of peaks
        
    '''
  
    sigma = 2. #width of Gaussian peaks

    #creation of arrays X,Y for constructing Gaussian array
    x = np.arange(shape[1])
    y = np.arange(shape[0])

    x = x-(len(x)-1)/2. #shift center of image to (0,0)
    y = y-(len(y)-1)/2.
    
    X,Y = np.meshgrid(x,y)
    
    kernel = np.zeros(shape)
    
    #for a 2D lattice we need two generator vectors that we can use to make all the points
    a1 = spacing*np.array([1,0])
    a2 = spacing*np.array([np.cos(np.pi/3.), np.sin(np.pi/3.)])
    
    #apply a rotation to the primative vectors
    angle = orientation*np.pi/180. #convert angle to radians
    rot_M = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]) #2d rotation matrix
    
    a1 = rot_M.dot(a1)
    a2 = rot_M.dot(a2)
    
    def G2D(x,y,x0,y0,s):
        #function to make 2d gaussian
        return np.exp( -((x-x0)**2 + (y-y0)**2)/ s*s)
    
    num = 0  #counts how many points are in the kernel, for normalization purposes
    
    points = []
    
    #generate a bunch of lattice points
    for i in np.arange(-12,13):
        for j in np.arange(-12,13):
            aij = a1*i+a2*j
            
            if aij[0]<max(x) and aij[0]>min(x) and aij[1]<max(y) and aij[1]>min(y):
                #Only modify points within 3 sigma of the center, reduces computation time
                cut = np.where((np.abs(X-aij[0])<3*sigma)&(np.abs(Y-aij[1])<3*sigma))

                #make a lattice point by having a bunch of gaussians            
                lat_point = G2D(X[cut],Y[cut],aij[0],aij[1],sigma)

                kernel[cut] += lat_point
                num+=1
                
                aij[0] += (len(x)-1)/2. #bring center back to image center
                aij[1] += (len(y)-1)/2.
                
                points.append(aij)

    return kernel/float(num), np.array(points)

##################################

def append_kernel_to_df(a,theta, df):
    '''
    This function appends the kernels of size (101,101) and (201,201) to a 
    database of kernel objects for use in later analysis.

    Parameters
    ----------
    a : float
        lattice spacing for kernel
    theta : float
        orientation angle for kernel
    df : pandas Dataframe
        database object that we want to add the kernel to

    Returns
    -------
    df : pandas Dataframe
        copy of df with the new kernel added to it

    '''
    
    #will have two different shapes of kernels (101,101), (201,201)
    
    kernel_101, points = make_triang_kernel(a,theta,(101,101))
    kernel_201, points = make_triang_kernel(a,theta,(201,201))
    
    df = df.append({'a':a, 'ang':theta, 'kernel_101':kernel_101, 'kernel_201':kernel_201},ignore_index=True)
    
    return df

##################################

def get_kernel_a(a):
    '''
    Looks at a folder with many .pkl files of kernels and returns the dataframe with 
    the given lattice spacing

    Parameters
    ----------
    a : float
        lattice spacing to extract        

    Returns
    -------
    df : pandas Dataframe
        database of kernels at lattice spacing a for a variety of angle orientations

    '''
    #src = './kernels/'
    src = 'C:/Users/tevid/Desktop/DNA_Origami/Code/TubeAnalysis/kernels/'
    filename = 'fft_kernel_a'+str(a)+'.pkl'
    
    df = pd.read_pickle(src+filename)
    
    return df

##################################

def analyze_a_kernels(a, fft, ints):
    '''
    This applies the desired kernel to the FFT array from the tube structure. This returns
    the Sum( kernel * FFT ), think of as a convolution

    Parameters
    ----------
    a : float
        lattice spacing to analyze
    fft : ndarray
        (N,N) array of the FFT of a tube image
    ints : list
        list of convolution values to append results to

    Returns
    -------
    ints : list
        returns updated list of convolution values

    '''

    df = get_kernel_a(a)
    orients = np.unique(df['ang'])

    for ang in orients:

        kernel_ang = df['kernel_201'][df['ang']==ang].to_numpy()[0][0]
        #print(kernel_ang)

        integral = np.sum(fft.flatten()*kernel_ang.flatten())

        ints.append([a,ang,integral])


    return ints

##################################

def get_fft(image):
    """
    Takes the FFT of the input image. It will also remove FFT values at the center of the image

    Parameters
    ----------
    image : ndarray
        image of a tube

    Returns
    -------
    p_spec : ndarray
        fft of the input image

    """
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    p_spec = np.abs(fshift[:,::-1])
    
    loc_center = np.where(p_spec==p_spec.max())
    
    center_x = int(loc_center[0])
    center_y = int(loc_center[1])
    
    x = np.arange(np.shape(image)[0])-center_x
    y = np.arange(np.shape(image)[1])-center_y
    
    X, Y = np.meshgrid(x,y)
    
    center_cut = np.where(X**2+Y**2<10**2) #sets pixels within radius 10 of the center to 0
    
    p_spec[center_cut]=0
        
    size = 100
    
    return p_spec[int(center_x-size):int(center_x+size+1), int(center_y-size):int(center_y+size+1)]
   
##################################

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Used in the fitting of 2d data to a Gaussian peak. Input to scipy.optimize.curve_fit.
    Note that for curve_fit the function input is expected to return a 1d array.
    """

    #amplitude, xo, yo, sigma_x, sigma_y, theta, offset = params
    (x,y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g
   
##################################

def fit_gaussian(ints):
    """
    Given a list of values with x,y coordinates, fits a 2d gaussian to the values

    Parameters
    ----------
    ints : ndarry
        array that has shape (3,N) where each entry is (x,y,z) data

    Returns
    -------
    popt : list
        fit parameters for the 2d gaussian fit to the input
    pcov : ndarray
        covariance matrix for the fit parameters

    """
    
    #extract list of the x, y, z data
    ints_x = ints.T[0]
    ints_y = ints.T[1]
    ints_z = ints.T[2]
    
    ############## TO DO ###############
    #need to account for the if the peak is located on the edge of the angle spectra
    #we will just shift the peak to the center of the angle values
    #ang_loc_max = np.where(ints_z==(ints_z).max())[1]
    #ang_shift = int(len(orients)/2.-ang_loc_max[0])
    #ints_z = np.roll(ints_z,ang_shift,axis=1)
    
    loc_max = np.where(ints_z==(ints_z).max())
    
    #get an estimate of the location of the peak and initial guess values
    average = np.average(ints_z)
    maximum = ints_z[loc_max][0]
    
    loc_max = np.where(ints_z==(ints_z).max())
    x0, y0 = ints_x[loc_max][0], ints_y[loc_max][0]
    
    p0 = [maximum-average, x0, y0, 0.1, 1.0, 0, average] #initial guess for parameters
    
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (ints_x,ints_y), ints_z-average, p0=p0)
    
    return popt, pcov
   
##################################   
   
def fwhm_width(ints,shape_a_o, da, do):
    """
    This provides an estimate of the uncertainties of the peak width by looking at
    the size of the area above the FWHM value of the peak

    Parameters
    ----------
    ints : ndarry
        array that has shape (3,N) where each entry is (x,y,z) data
    shape_a_o : tuple
        conatins the number of unique x and y values, assumes that z data is on a
        uniform square lattice of x,y points
    da : float
        spacing of x data
    do : float
        spacing of y data

    Returns
    -------
    a_peak : float
        x location of peak
    a_width : float
        x width of peak
    o_peak : float
        y location of peak
    o_width : float
        y width of peak

    """
    ints_z = ints.T[2].reshape(shape_a_o)
    
    ints_a = ints.T[0]
    ints_o = ints.T[1]
    ints_z_flat = ints.T[2]
    
    loc_max_flat = np.where(ints_z_flat==ints_z_flat.max())[0][0]
    a_peak, o_peak = ints_a[loc_max_flat], ints_o[loc_max_flat]
    
    #need to account for the if the peak is located on the edge of the angle spectra
    #we will just shift the peak to the center of the angle values
    ang_loc_max = np.where(ints_z==(ints_z).max())[1]
    #ang_shift = int(len(orients)/2.-ang_loc_max[0])
    ang_shift = int(len(ints_o)/2.-ang_loc_max[0]) 
    
    ints_z = np.roll(ints_z,ang_shift,axis=1)

    #threshold image based on the hieght of half of the peak
    bkgd, peak_val = np.average(ints_z), ints_z.max()
    thres_ints_z = threshold_image(ints_z,invert=False, auto=False, auto_thres=peak_val/2.+ bkgd/2.)
    
    thres_ints_z = return_largest_area(thres_ints_z)
    
    #to find the size of the thresholded region, look for edges of the blob
    loc_blob = np.where(thres_ints_z>0.) 
    
    a_width = (max(loc_blob[0]) - min(loc_blob[0]))*da
    o_width = (max(loc_blob[1]) - min(loc_blob[1]))*do

    fwhm_to_sigma = 2*np.sqrt(2*np.log(2))

    a_width = a_width/fwhm_to_sigma
    o_width = o_width/fwhm_to_sigma
    
    return a_peak, a_width, o_peak, o_width

##################################

def threshold_image(image,invert=False, auto=True, auto_thres=100.):
    """
    Thresholds the input image either from the isodata method or from an input
    threshold value

    Parameters
    ----------
    image : ndarray
        input image
    invert : bool, optional
        If False then values above the threshold are set to 255 and values below
        to 0. If True then values above the threshold are set to 0 and values below
        to 255. The default is False.
    auto : bool, optional
        If True the method defaults to isodata method. If False then the method
        will use an input value for the threshold. The default is True.
    auto_thres : float, optional
        When using auto=False this is the value of the threshold. The default is 100.

    Returns
    -------
    thres_img : ndarray
        thresholded image with values of 0 and 255.

    """
    if auto:
        ''' thresholds image off isodata method'''
        threshold_value = skimage.filters.threshold_isodata(image)
    else: threshold_value = auto_thres
    
    if not invert:
        thres_img = np.zeros(np.shape(image))
        thres_img[image>threshold_value]=255
    if invert:
        thres_img = np.zeros(np.shape(image)) + 255
        thres_img[image>threshold_value]=0
        
    return thres_img
   
################################## 
   
def invert_image(image):
    """
    Assumes an input image is thresholded and binary. Inverts high and low values 
    of the image.

    Parameters
    ----------
    image : ndarray
        binary image input

    Returns
    -------
    inverted : ndarray
        inverted image output

    """
    #if you have a binarized image this returns the invert with values of 255 and 0
    inverted = np.zeros(np.shape(image))
    inverted[image==0]=255
    inverted[image>0]=0
    return inverted
   
##################################

def find_image_axes(image):
    """
    This method finds the axes of the image by calculating
    the central moments of the image. It then calculates the 
    eigen vectors and values of the covariance matrix. The angles
    formed by the eigen vectors are the axes of the image.

    Parameters
    ----------
    image : ndarray
        input image of tubes

    Returns
    -------
    major_axis : float
        major axis of the image
    minor_axis : float
        minor axis of the image
    M : ndarray
        array of the masses used to compute the axes, this is after different
        cuts and thresholds have been applied

    """
    #smooth input image
    s_image = ndim.filters.gaussian_filter(image,2)
    
    com = measurements.center_of_mass(s_image)
    
    x = np.arange(np.shape(image)[1])
    y = np.arange(np.shape(image)[0])

    X,Y = np.meshgrid(x,y)
    X = X-com[1]
    Y = Y-com[0]
    
    #returns largest region after thresholding
    thres_image = threshold_image(s_image, invert=False)
    thres_image = invert_image(return_largest_area(invert_image(thres_image)))
    
    M = thres_image
    
    #only considers mass of image in a circular region the purpose of this is to
    #remove effects of having tubes extended out of the frame of the image, this
    #can bias the measured axes.
    loc_circ = np.where(X**2+Y**2> (np.shape(image)[1]/2.)**2)
    M[loc_circ]=255
    
    M = invert_image(return_largest_area(M))
    
    xs, ys, ms = X.flatten(), Y.flatten(), M.flatten()

    def mu_pq(p,q,x,y,m):
        #compue the moment mu(p,q) of image
        return np.sum(m*np.power(x,p)*np.power(y,q))

    #these the are relevant central moments of the image
    mu_20 = mu_pq(2,0,xs,ys,ms)/mu_pq(0,0,xs,ys,ms)
    mu_02 = mu_pq(0,2,xs,ys,ms)/mu_pq(0,0,xs,ys,ms)
    mu_11 = mu_pq(1,1,xs,ys,ms)/mu_pq(0,0,xs,ys,ms)

    #covariance matrix
    I = np.array([[mu_20, mu_11]
                 ,[mu_11, mu_02]])

    eig_val, eig_vec = np.linalg.eig(I)

    #orders major vs minor off which eigenvalue is larger
    major_axis_index = np.where(eig_val==max(eig_val))[0][0]
    minor_axis_index = np.where(eig_val==min(eig_val))[0][0]

    #computes angle of the eigenvectors and converts to degrees
    major_axis = np.arctan2(eig_vec[major_axis_index][1],eig_vec[major_axis_index][0])*180/np.pi
    minor_axis = np.arctan2(eig_vec[minor_axis_index][1],eig_vec[minor_axis_index][0])*180/np.pi

    #reformats angles to be in the right-half plane
    if major_axis>90: major_axis-=180
    if major_axis<-90: major_axis+=180
    if minor_axis>90: minor_axis-=180
    if minor_axis<-90: minor_axis+=180
    
    return major_axis, minor_axis, M

##################################

def return_largest_area(thres_img):
    """
    This finds the largest connected region within a thersholded image and outputs
    an image with only that section as non-zero

    Parameters
    ----------
    thres_img : ndarray
        takes in a binary image of thresholded image

    Returns
    -------
    thres_img : ndarray
        only the largest connected area is returned

    """
    
    #next need to make sure there are no suprious peaks in the image
    label_im, num_labels = ndim.label(thres_img) #labels connected regions of the image
    
    sizes = []
    for i in range(num_labels):
        sizes.append(np.sum(label_im[label_im==i+1])/float(i+1)) #counts number of pixels for each label
    
    if num_labels>1:
        label_big = np.where(sizes==max(sizes))[0][0]+1
        label_im[label_im != label_big]=0
        thres_img = label_im/float(label_big)
    
    return thres_img

##################################

def sides_location(thres_img, radius):
    """
    Returns the x,y coordinates of the left and right hand sides of the rotated
    thresholded image. These points can then be fit to a line or used in some way
    to correct mistakes in the tube angle acquisition.

    Parameters
    ----------
    thres_img : ndarray
        assumes that the thresholded image input has been rotated to a close to
        vertical position
    radius : float
        radius used in the central moment calculation

    Returns
    -------
    left_side : ndarray
        array of shape (2,N) where the rows are the x and y coordinates of the
        left side positions. i.e. x positions called as left_side[0].
    right_side : ndarray
        array of shape (2,N) where the rows are the x and y coordinates of the
        right side positions. i.e. x positions called as left_side[0].

    """
    
    com = measurements.center_of_mass(thres_img)
    
    left_side, right_side = [], []
    
    #print(np.unique(thres_img))
    
    for i in range(np.shape(thres_img)[0]):
        tube_loc = np.where(thres_img[i,:]>0)[0]
        
        if len(tube_loc)!=0:        
            left, right = tube_loc[0], tube_loc[-1]
            if (i-com[1])**2 + (left-com[0])**2 < (radius*.93)**2:
                left_side.append([i,left])
            if (i-com[1])**2 + (right-com[0])**2 < (radius*.93)**2:
                right_side.append([i,right])
        
    left_side, right_side = np.array(left_side).T, np.array(right_side).T
    
    #fit_l = np.polyfit(left_side.T[0], left_side.T[1], deg=1)
    #fit_r = np.polyfit(right_side.T[0], right_side.T[1], deg=1)
    
    #print(fit_l, fit_r)
    #print('line angles for left (right):', 180*np.arctan2(fit_l[0],1)/np.pi , 180*np.arctan2(fit_r[0],1)/np.pi )

    return left_side, right_side
    
##################################

if __name__ == "__main__":
    '''
    This is an example of how to use some of the functions in here.
    '''
    
    src = 'C:/Users/tevid/Desktop/DNA_Origami/Code/TubeAnalysis/'
    
    img_fn = 'TubeSection2.png'
    #img_fn = '27.png'
    #img_fn = '6.png'
    image = pims.ImageSequence(src+img_fn)[0]
    
    '''
    first we find the orientation and width of the tube
    '''
    
    smooth_img = ndim.filters.gaussian_filter(image,4)
    
    major_axis, minor_axis, thres_img = find_image_axes(image)
    print('major and minor axis of image are ', major_axis, minor_axis)
    
    rotate_img = ndim.rotate(thres_img,-minor_axis, reshape=1)
    rotate_img = threshold_image(rotate_img)

    left, right = sides_location(rotate_img, np.shape(thres_img)[1]/2.)
    
    tube_thickness = np.average(rotate_img,axis=0)

    thres_thick = 100
    
    width = np.where(tube_thickness>thres_thick)[0][-1] - np.where(tube_thickness>thres_thick)[0][0]
    print('tube width is ', width)
    
    '''
    The following section finds the orientation of the triangular
    lattice on the tube.
    '''
    
    fft = get_fft(image)
    
    orients = np.linspace(0,59.5, num=60*2)
    a_s = np.linspace(10.0, 30.0, num = 20*4+1)
    
    da, do = a_s[1]-a_s[0], orients[1]-orients[0]
    
    ints = []

    for a in a_s:
        ints = analyze_a_kernels(a,fft,ints)
        
    ints = np.array(ints)
    
    shape_a_o = (len(a_s),len(orients))
    popt, pcov = fit_gaussian(ints)
    
    print('fit parameters from Gaussian fit:')
    print('lattice spacing: ', popt[1])
    print('lattice spacing error: ', popt[3])
    print('lattice angle: ', popt[2])
    print('lattice angle error: ', popt[4])
    
    a_peak, a_width, o_peak, o_width = fwhm_width(ints, shape_a_o, da, do)
    print('fit parameters from FWHM:')
    print('lattice spacing: ', a_peak)
    print('lattice spacing error', a_width)
    print('lattice angle: ', o_peak)
    print('lattice angle error', o_width)
    
    '''
    plot results from the analysis
    '''
    
    plt.subplot(321)
    plt.imshow(smooth_img)
    
    plt.subplot(322)
    plt.imshow(fft)
    
    plt.subplot(323)
    plt.imshow(rotate_img)
    plt.scatter(left[1],left[0],c='r', s=10)
    plt.scatter(right[1],right[0],c='b', s=10)
    
    plt.subplot(324)
    plt.plot(tube_thickness)
    
    plt.subplot(325)
    plt.tripcolor(ints.T[0], ints.T[1], ints.T[2])
    plt.scatter(a_peak,o_peak,s=50,facecolors='none',edgecolors='k')
    
    plt.show()
    