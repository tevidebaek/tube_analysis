# tube_analysis
For use of analyzing the structure of assembly discrete tubes.

The file tube_analysis.py is a collection of functions that can be used in the analysis of tubes. The "__main__" of this script is an example of how to use the various functions if you have TEM images of tubes to start with. A critical part of the analysis uses pre-generated .pkl files that contain refernce arrays. The end-user will have to generate their own .pkl files to use for analysis since there is a file limit on Git Hub. If you are a member of the Rogers lab there is a storage location for these .pkl files that you can transfer to your work folder.

# gui_analysis
Creates an interactive window to run analysis on your images.

To use this code you will need a folder that contains "tube_anlaysis.py", "./kernels/", "./data/", "./images/". "./kernels/" contains the reference arrays for the lattice comparison. "./images/" should contain the collection of images that you want to do analysis on. "./data/" will store the output .pkl file from the analysis.

For each image in "./images" the script will produce a GUI from which you can analyze the tube. We use Tkinter to create the interactive window. This window has pull down menus for "File" and "Analysis". The "File" menu has the options "Save" and "Exit". "Save" takes the stored values from the analysis of your image and apends them to a DataFrame file, here stored as a .pkl. "Exit" closes the window, after which a new one will open with the next image in the image folder. The "Analysis" contains three options which you should use in sequence:

"Region of Interest"  - allows the user to place three marks on the image that should be placed on the sides of the tube. Be sure to do it in order of two on one side of the tube and the third on the opposing side. Be sure to place the first two marks on the same side of the tube.

"Get Tube Parameters" - calculates the width and angle of the tube based on the three points chosen with the "Region of Interest" command. These values are stored and will be saved with the "Save" command.

"Lattice Angle - FFT" - this runs the kernel comparison to the FFT of the image to get the lattice angle. The angle and lattice spacing, with corresponding error are stored and will be saved with the "Save" command.
