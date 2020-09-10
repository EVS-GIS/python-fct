# -*- coding: utf-8 -*-
"""
This file contains are a series of helper functions that interact with
LSDTopoTools spatial data.

Written by Simon M. Mudd and Fiona J Clubb at the University of Edinburgh
Released under GPL3

22/06/2017
"""

#==============================================================================
def MapFigureSizer(figure_width_inches,aspect_ratio, cbar_loc = "None", title = "None",
                   cbar_width = 0.2,
                   cbar_text_width = 0.4,
                   cbar_padding = 0.1,
                   cbar_fraction = 1,
                   whitespace_padding = 0.2,
                   map_text_width = 0.65,
                   map_text_height = 0.45,
                   title_height=0.2):
    """This function takes a string size argument and calculates the size of the
    various components of a plot based on a map image making up the centre of the
    figure.

    We use inches because bloody yanks wrote matplotlib and figures in matplotlib use inches.
    Luckily we do not have to calculate bushels or furlongs of pixels somewhere,
    and the inches stupidity is somewhat limited.

    Args:
        figure_width_inches (flt): The figure width in inches
        aspect_ratio (flt): The width to height ratio of the data
        cbar_loc (string): the location of the colourbar, either "left", "right", "top",
        "bottom", or "none"
        title (bool): if true then adjust the height of the figure to make space for a title
        cbar_width (flt): the width of the colorbar
        text_padding (list): the padding around the map from the whitespace and the axis/tick labels

    Author: SMM
    """


    # The text padding list holds the height and width, in inches of
    # [0] = left/right tick marks+tick labels
    # [1] = left/right text (i.e., and exis label)
    # [2] = top/bottom tick marks+tick labels
    # [3] = top/bottom text (i.e., and exis label)
    # [4] = whitespace at the edge of the figure + between map and colourbar
    # [5] = colorbar text (e.g. tick labels+axis label)


    # This gets returned, we use it to make the figure.

    #======================================================

    # Now we need to figure out where the axis are. Sadly this requires
    # a load of tedious conditional statments about the location of the axes

    #Will's changes:
    #1/ Make space for the colourbar label on the left
    #2/ Double the cbar_padding to leave space for the colourbar values on the right.
    # NB: this should later be a function of font size
    #3/ Changed rotation of colourbar text to 90 and the labelpad to -75 in PlottingRaster.py

    if cbar_loc == "left":
        cbar_left_inches = whitespace_padding + cbar_text_width
        #cbar_left_inches = whitespace_padding
        map_left_inches = cbar_left_inches+cbar_width+map_text_width + 2*cbar_padding
        #map_left_inches = cbar_left_inches+cbar_width+cbar_text_width+map_text_width+cbar_padding
        map_width_inches = figure_width_inches-map_left_inches-whitespace_padding
        map_height_inches = map_width_inches/aspect_ratio



        map_bottom_inches = whitespace_padding+map_text_height
        cbar_bottom_inches = map_bottom_inches
        figure_height_inches = map_bottom_inches+map_height_inches+whitespace_padding
        if title != "None":
            # add some space for a title if needed. At the moment this is hard coded but
            # we might want to change this for the font size.
            figure_height_inches = figure_height_inches+title_height

        fig_size_inches = [figure_width_inches,figure_height_inches]

        # print("cbar_left: "+str(cbar_left_inches)+" map left: "+str(map_left_inches))
        # print("cbar_bottom: "+str(cbar_bottom_inches)+ " map bottom: "+str(map_bottom_inches))

        map_axes = [map_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    map_width_inches/figure_width_inches,
                    map_height_inches/figure_height_inches]
        cbar_axes = [cbar_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    cbar_width/figure_width_inches,
                    cbar_fraction*(map_height_inches/figure_height_inches)]

    elif cbar_loc == "right":

        map_left_inches = whitespace_padding+map_text_width
        cbar_left_inches= figure_width_inches-whitespace_padding-cbar_width-cbar_text_width
        map_right_inches = cbar_left_inches-cbar_padding

        map_width_inches = map_right_inches-map_left_inches
        map_height_inches = map_width_inches/aspect_ratio

        map_bottom_inches = whitespace_padding+map_text_height
        cbar_bottom_inches = map_bottom_inches
        figure_height_inches = map_bottom_inches+map_height_inches+whitespace_padding
        if title != "None":
            # add some space for a title if needed. At the moment this is hard coded but
            # we might want to change this for the font size.
            figure_height_inches = figure_height_inches+title_height

        fig_size_inches = [figure_width_inches,figure_height_inches]

        # print("cbar_left: "+str(cbar_left_inches)+" map left: "+str(map_left_inches))
        # print("cbar_bottom: "+str(cbar_bottom_inches)+ " map bottom: "+str(map_bottom_inches))


        map_axes = [map_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    map_width_inches/figure_width_inches,
                    map_height_inches/figure_height_inches]
        cbar_axes = [cbar_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    cbar_width/figure_width_inches,
                    cbar_fraction*(map_height_inches/figure_height_inches)]

    elif cbar_loc == "top":
        print("I am placing the colourbar on the top")

        map_left_inches = whitespace_padding+map_text_width
        map_right_inches = figure_width_inches-whitespace_padding
        map_width_inches = map_right_inches-map_left_inches
        map_height_inches = map_width_inches/aspect_ratio

        cbar_left_inches= map_left_inches

        map_bottom_inches = whitespace_padding+map_text_height
        cbar_bottom_inches = map_bottom_inches+map_height_inches+cbar_padding+cbar_text_width

        figure_height_inches = cbar_bottom_inches+cbar_width+whitespace_padding
        if title != "None":
            # add some space for a title if needed. At the moment this is hard coded but
            # we might want to change this for the font size.
            title_height = 0.5
            figure_height_inches = figure_height_inches+title_height

        fig_size_inches = [figure_width_inches,figure_height_inches]

        # print("cbar_left: "+str(cbar_left_inches)+" map left: "+str(map_left_inches))
        # print("cbar_bottom: "+str(cbar_bottom_inches)+ " map bottom: "+str(map_bottom_inches))


        map_axes = [map_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    map_width_inches/figure_width_inches,
                    map_height_inches/figure_height_inches]
        cbar_axes = [cbar_left_inches/figure_width_inches,
                    cbar_bottom_inches/figure_height_inches,
                    cbar_fraction*(map_width_inches/figure_width_inches),
                    cbar_width/figure_height_inches]

    elif cbar_loc == "bottom":
        print("I am placing the colourbar on the bottom")

        map_left_inches = whitespace_padding+map_text_width
        map_right_inches = figure_width_inches-whitespace_padding
        map_width_inches = map_right_inches-map_left_inches
        map_height_inches = map_width_inches/aspect_ratio

        cbar_left_inches= map_left_inches

        cbar_bottom_inches = whitespace_padding+cbar_text_width
        map_bottom_inches = cbar_bottom_inches+cbar_width+cbar_padding+map_text_height

        whitespace_padding+map_text_height

        figure_height_inches = map_bottom_inches+map_height_inches+whitespace_padding
        if title != "None":
            # add some space for a title if needed. At the moment this is hard coded but
            # we might want to change this for the font size.
            figure_height_inches = figure_height_inches+title_height

        fig_size_inches = [figure_width_inches,figure_height_inches]

        # print("cbar_left: "+str(cbar_left_inches)+" map left: "+str(map_left_inches))
        # print("cbar_bottom: "+str(cbar_bottom_inches)+ " map bottom: "+str(map_bottom_inches))


        map_axes = [map_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    map_width_inches/figure_width_inches,
                    map_height_inches/figure_height_inches]
        cbar_axes = [cbar_left_inches/figure_width_inches,
                    cbar_bottom_inches/figure_height_inches,
                    cbar_fraction*(map_width_inches/figure_width_inches),
                    cbar_width/figure_height_inches]

    else:
        print("No colourbar")

        map_left_inches = whitespace_padding+map_text_width
        map_right_inches = figure_width_inches-whitespace_padding
        map_width_inches = map_right_inches-map_left_inches
        map_height_inches = map_width_inches/aspect_ratio

        map_bottom_inches = whitespace_padding+map_text_height

        figure_height_inches = map_bottom_inches+map_height_inches+whitespace_padding
        if title != "None":
            # add some space for a title if needed. At the moment this is hard coded but
            # we might want to change this for the font size.
            figure_height_inches = figure_height_inches+title_height

        fig_size_inches = [figure_width_inches,figure_height_inches]

        map_axes = [map_left_inches/figure_width_inches,
                    map_bottom_inches/figure_height_inches,
                    map_width_inches/figure_width_inches,
                    map_height_inches/figure_height_inches]
        cbar_axes = None

    # print("The figure size is: ")
    # print(fig_size_inches)
    # print("Map axes are:")
    # print(map_axes)
    # print("cbar_axes are:")
    # print(cbar_axes)
    return fig_size_inches, map_axes, cbar_axes