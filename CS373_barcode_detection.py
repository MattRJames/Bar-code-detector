# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# You can add your own functions here:
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def greyScale(red,green,blue,image_width,image_height):
    #step 1 convert to grey scale
    grey_scale = createInitializedGreyscalePixelArray(image_width, image_height)
    # parse through every pixel and place
    for i in range(0,image_height):
        for j in range(0,image_width):
            grey_scale[i][j] = 0.299*red[i][j] +0.587*green[i][j]+0.114*blue[i][j]

    return grey_scale

def normalize(pixel_array,image_width,image_height):
    normalized = createInitializedGreyscalePixelArray(image_width, image_height)

    f_min , f_max = 256,-1
    #find f min and f max
    for row in pixel_array:
        min_value = min(row)
        max_value = max(row)
        if min_value < f_min:
            f_min = min_value
        if max_value>f_max:
            f_max = max_value

    #use formula with gmax = 255 and gmin = 0 and change value of each pixel
    g_max = 255
    g_min = 0

    for i in range(0,image_height):
        for j in range(0,image_width):
            normalized[i][j] = (pixel_array[i][j] - f_min) * ((g_max-g_min)/(f_max-f_min)) + g_min
    return normalized


def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):
            if (i == 0) or (j == 0) or (i == image_height - 1) or (j == image_width - 1):
                output[i][j] = 0.0

            else:
                total = 0
                # top right
                topRight = pixel_array[i - 1][j + 1]

                # middle right
                middleRight = total + 2 * (pixel_array[i][j + 1])

                # bottom right
                bottomRight = total + pixel_array[i + 1][j + 1]

                # top left
                topLeft = total + (-1) * pixel_array[i - 1][j - 1]
                # middle left
                middleLeft = total + -2 * (pixel_array[i][j - 1])
                # bottom left
                bottomLeft = total + (-1) * pixel_array[i + 1][j - 1]
                output[i][j] = abs((topRight + middleRight + bottomRight + topLeft + middleLeft + bottomLeft) / 8.0)

    return output


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(0, image_height):
        for j in range(0, image_width):
            if (i == 0) or (j == 0) or (i == image_height - 1) or (j == image_width - 1):
                output[i][j] = 0.0

            else:
                total = 0
                # top right
                topRight = pixel_array[i - 1][j + 1]

                # middle right
                topMiddle = 2 * (pixel_array[i - 1][j])

                # bottom right
                bottomRight = -1 * pixel_array[i + 1][j + 1]

                # top left
                topLeft = (1) * pixel_array[i - 1][j - 1]
                # middle left
                bottomMiddle = -2 * (pixel_array[i + 1][j])
                # bottom left
                bottomLeft = (-1) * pixel_array[i + 1][j - 1]
                output[i][j] = abs((topRight + topMiddle + bottomRight + topLeft + bottomMiddle + bottomLeft) / 8.0)

    return output

def calc_abs_difference(horizontal_sobel,vertical_sobel,image_width,image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(0,image_height):
        for j in range(0,image_width):
            output[i][j] = abs(horizontal_sobel[i][j]-vertical_sobel[i][j])


    return output


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):

            # top left corner
            if (i == 0) & (j == 0):
                topLeft = pixel_array[i][j]
            elif i == 0:
                topLeft = pixel_array[i][j - 1]
            elif j == 0:
                topLeft = pixel_array[i - 1][j]
            else:
                topLeft = pixel_array[i - 1][j - 1]

            # top right corner
            if (i == 0) & (j == image_width - 1):
                topRight = pixel_array[i][j]
            elif i == 0:
                topRight = pixel_array[i][j + 1]
            elif j == image_width - 1:
                topRight = pixel_array[i - 1][j]
            else:
                topRight = pixel_array[i - 1][j + 1]

            # bottom Left corner
            if (i == image_height - 1) & (j == 0):
                bottomLeft = pixel_array[i][j]
            elif i == image_height - 1:
                bottomLeft = pixel_array[i][j - 1]
            elif j == 0:
                bottomLeft = pixel_array[i + 1][j]
            else:
                bottomLeft = pixel_array[i + 1][j - 1]

            # bottom Right corner
            if (i == image_height - 1) & (j == image_width - 1):
                bottomRight = pixel_array[i][j]
            elif i == image_height - 1:
                bottomRight = pixel_array[i][j + 1]
            elif j == image_width - 1:
                bottomRight = pixel_array[i + 1][j]
            else:
                bottomRight = pixel_array[i + 1][j + 1]

            # top middle
            if i == 0:
                topMiddle = 2 * pixel_array[i][j]
            else:
                topMiddle = 2 * pixel_array[i - 1][j]

            # bottom middle
            if i == image_height - 1:
                bottomMiddle = 2 * pixel_array[i][j]
            else:
                bottomMiddle = 2 * pixel_array[i + 1][j]
            # middle left
            if j == 0:
                middleLeft = 2 * pixel_array[i][j]
            else:
                middleLeft = 2 * pixel_array[i][j - 1]

            # middle right
            if j == image_width - 1:
                middleRight = 2 * pixel_array[i][j]
            else:
                middleRight = 2 * pixel_array[i][j + 1]

            middle = 4 * pixel_array[i][j]
            output[i][j] = abs((
                                           middle + topMiddle + bottomMiddle + topRight + middleRight + bottomRight + topLeft + middleLeft + bottomLeft) / 16.0)

    return output


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    blank = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(0, len(pixel_array)):
        for j in range(0, len(pixel_array[i])):
            if pixel_array[i][j] < threshold_value:
                blank[i][j] = 0
            else:
                blank[i][j] = 255

    return blank

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):
            values = []
            for yoffset in range(-1, 2):
                for xoffset in range(-1, 2):

                    if (i + yoffset < 0) or (i + yoffset > image_height - 1) or (j + xoffset < 0) or (
                            j + xoffset > image_width - 1):
                        values.append(0)

                    else:
                        values.append(pixel_array[i + yoffset][j + xoffset])

            if values.count(0) != 9:
                output[i][j] = 255
    return output
def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):
            values = []
            for yoffset in range(-2, 3):
                for xoffset in range(-2, 3):

                    if (i + yoffset < 0) or (i + yoffset > image_height - 1) or (j + xoffset < 0) or (
                            j + xoffset > image_width - 1):
                        values.append(0)

                    else:
                        values.append(pixel_array[i + yoffset][j + xoffset])

            if values.count(0) != 25:
                output[i][j] = 255
    return output

def computeDilation8Nbh7x7FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):
            values = []
            for yoffset in range(-3, 4):
                for xoffset in range(-3, 4):

                    if (i + yoffset < 0) or (i + yoffset > image_height - 1) or (j + xoffset < 0) or (
                            j + xoffset > image_width - 1):
                        values.append(0)

                    else:
                        values.append(pixel_array[i + yoffset][j + xoffset])

            if values.count(0) != 49:
                output[i][j] = 255
    return output


def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):
            values = []
            for yoffset in range(-2, 3):
                for xoffset in range(-2, 3):

                    if (i + yoffset < 0) or (i + yoffset > image_height - 1) or (j + xoffset < 0) or (
                            j + xoffset > image_width - 1):
                        values.append(0)

                    else:
                        values.append(pixel_array[i + yoffset][j + xoffset])

            if values.count(0) == 0:
                output[i][j] = 255

    return output

def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(0, image_height):
        for j in range(0, image_width):
            values = []
            for yoffset in range(-1, 2):
                for xoffset in range(-1, 2):

                    if (i + yoffset < 0) or (i + yoffset > image_height - 1) or (j + xoffset < 0) or (
                            j + xoffset > image_width - 1):
                        values.append(0)

                    else:
                        values.append(pixel_array[i + yoffset][j + xoffset])

            if values.count(0) == 0:
                output[i][j] = 255

    return output


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    label_dict = {}
    label_id = 1
    for i in range(0, image_height - 1):
        for j in range(0, image_width):
            if (pixel_array[i][j] != 0) & (output[i][j] == 0):
                q = Queue()
                q.enqueue([i, j])
                label_dict[label_id] = 0
                output[i][j] = label_id
                while (q.isEmpty() == False):
                    p = q.dequeue()
                    label_dict[label_id] = label_dict[label_id] + 1
                    output[p[0]][p[1]] = label_id
                    if p[0] != 0:
                        if (pixel_array[p[0] - 1][p[1]] > 0) & (output[p[0] - 1][p[1]] == 0):
                            q.enqueue([p[0] - 1, p[1]])
                            output[p[0] - 1][p[1]] = label_id

                    if p[0] != image_height - 1:
                        if (pixel_array[p[0] + 1][p[1]] > 0) & (output[p[0] + 1][p[1]] == 0):
                            q.enqueue([p[0] + 1, p[1]])
                            output[p[0] + 1][p[1]] = label_id

                    if p[1] != 0:
                        if (pixel_array[p[0]][p[1] - 1] > 0) & (output[p[0]][p[1] - 1] == 0):
                            q.enqueue([p[0], p[1] - 1])
                            output[p[0]][p[1] - 1] = label_id

                    if p[1] != image_width - 1:
                        if (pixel_array[p[0]][p[1] + 1] > 0) & (output[p[0]][p[1] + 1] == 0):
                            q.enqueue([p[0], p[1] + 1])
                            output[p[0]][p[1] + 1] = label_id

                label_id = label_id + 1

    return (output, label_dict)

def calc_bounding_box(pixel_array,image_width,image_height,key):
    min_x,min_y,max_x,max_y = 100000,100000,-1,-1

    for i in range(0,image_height):
        for j in range(0,image_width):
            if pixel_array[i][j] == key:
                if i < min_y:
                    min_y=i
                elif i > max_y:
                    max_y=i
                if j < min_x:
                    min_x=j
                elif j > max_x:
                    max_x = j
    return min_x,min_y,max_x,max_y


def computeBoxAveraging5x5(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(0, image_height):
        for j in range(0, image_width):
            values = []
            for yoffset in range(-2, 3):
                for xoffset in range(-2, 3):

                    if (i + yoffset < 0) or (i + yoffset > image_height - 1) or (j + xoffset < 0) or (
                            j + xoffset > image_width - 1):
                        values.append(0)

                    else:
                        values.append(pixel_array[i + yoffset][j + xoffset])
            output[i][j] = sum(values)/len(values)

    return output


# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode1"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')





    # STUDENT IMPLEMENTATION here

    grey_scale = greyScale(px_array_r,px_array_g,px_array_b,image_width,image_height)
    print("converted to grey scale")
    normalized = normalize(grey_scale,image_width,image_height)
    print("normalized")

    vertical_sobel = computeVerticalEdgesSobelAbsolute(normalized, image_width, image_height)
    print("computed vetical sobel")
    horizontal_sobel = computeHorizontalEdgesSobelAbsolute(normalized, image_width, image_height)
    print("computed horizontal sobel")
    combined_edges = calc_abs_difference(horizontal_sobel,vertical_sobel,image_width,image_height)
    print("computed combined edges")

    holding = combined_edges
    for i in range(0,1):
        holding = computeBoxAveraging5x5(holding, image_width, image_height)
    print("filtering complete")

    gaussian_filtered = holding
    threshholded = computeThresholdGE(gaussian_filtered,20,image_width,image_height)

    holding = threshholded
    #start with a smaller erosion to remove small connections
    holding = computeErosion8Nbh3x3FlatSE(holding, image_width, image_height)
    holding = computeErosion8Nbh3x3FlatSE(holding, image_width, image_height)
    for i in range(0,2):
        holding = computeDilation8Nbh5x5FlatSE(holding,image_width,image_height)

    for i in range(0, 3):
        holding = computeErosion8Nbh5x5FlatSE(holding, image_width, image_height)

    for i in range(0,2):
        holding = computeDilation8Nbh7x7FlatSE(holding,image_width,image_height)
    holding = computeDilation8Nbh5x5FlatSE(holding, image_width, image_height)



    diluted_and_eroded = holding
    print("dilution and erosion complete")
    labelled_components = computeConnectedComponentLabeling(diluted_and_eroded,image_width,image_height)
    print("components identified")
    component_dict = labelled_components[1]
    image =labelled_components[0]

    #select the relevant area,largest won't always be right but minimum threshold required to filter out irrelevant regions
    candidates = []
    component_dict_copy = component_dict.copy()


    while (len(candidates) < 6) & (len(component_dict_copy) > 0):
        candidates.append(max(component_dict_copy,key=component_dict_copy.get))
        component_dict_copy.pop(max(component_dict_copy,key=component_dict_copy.get))

    drop = []
    for key in candidates:
        if component_dict[key] < 2400:
            drop.append(key)
    for key in drop:
        candidates.remove(key)




#want to find the region that has a sufficient density and has an aspect ratio around 1.8
    bbox_min_x,bbox_min_y,bbox_max_x,bbox_max_y = None,None,None,None

    for key in candidates:
        min_x,min_y,max_x,max_y = calc_bounding_box(image,image_width,image_height,key)
        width = max_x-min_x
        height = max_y-min_y
        aspect_ratio = width/height

        density = component_dict[key]/(width*height)

        if ((aspect_ratio <1.8)&(aspect_ratio >0.4)&(density > 0.7)):

            bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = min_x,min_y,max_x,max_y
            rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                             edgecolor='g', facecolor='none')
            axs1[1, 1].add_patch(rect)




    px_array = px_array_r


    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    try:
        rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    except:
        rect = Rectangle((0, 1), 1, 1, linewidth=1,
                         edgecolor='g', facecolor='none')

    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()