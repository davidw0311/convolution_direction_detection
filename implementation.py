import imageio

import numpy as np
import scipy.signal as sp

def correlate_adjacent_frames( previous_frame, current_frame ):
    ''' 
        This function takes in two NumPy arrays filled with uint8 (integers between 0 and 255 inclusive) of size 160 by 160 and returns a 
        NumPy array of uint8 that represents `previous_frame` being cross correlated with a convolutional kernel of size 110 by 110 created 
        by removing the first and last 25 pixels in each dimension from `current_frame`. 

        The array returned should be of size 110 by 110 but the elements that are dependent on values "outside" the provided pixels of 
        `previous_frame` should be set to zero.

        Before computing the cross-correlation, you should normalize the input arrays in the range 0 to 1 and then subtracting the mean pixel 
        value of both inputs (i.e. the mean value of the list created by concatenating all pixel intensities of `current_frame` and all pixel
        intensities of `previous_frame`)
    '''
    height, width = current_frame.shape

    normalized_previous_frame = np.copy(previous_frame) / 255
    normalized_current_frame = np.copy(current_frame) / 255

    average_pixel_value = (np.mean(normalized_previous_frame) + np.mean(normalized_current_frame))/2

    normalized_previous_frame -= np.full_like(normalized_previous_frame, average_pixel_value)
    normalized_current_frame -= np.full_like(normalized_current_frame, average_pixel_value)

    convolutional_kernel = np.copy(normalized_current_frame[25:height-25, 25:width-25])


    # output_image = sp.correlate2d(normalized_previous_frame, convolutional_kernel, mode='valid')
    output_image = np.zeros((50, 50))
    for r in range(50):
        for c in range(50):
            output_image[r][c] = np.sum(convolutional_kernel * normalized_previous_frame[r:r+110, c:c+110])

    max_value = output_image.max()
    min_value = output_image.min()
    output_image = (output_image - np.full_like(output_image, min_value)) * 255.0 / (max_value - min_value)
    
    padded_image = output_image.astype(np.uint8)
    # padded_image = np.zeros((110,110))
    # padded_image[0:50, 0:50] = output_image.astype(np.uint8)

    return padded_image

def make_correlation_video( input_filename, output_filename=None ):
    '''
        This function takes in an input filename string of a GIF and an optional output filename string. It should read the video from the input filename 
        using the mimread function from imageio and then apply correlate_adjacent_frames to each pair of adjacent frames 
        (i.e. between frame 0 and frame 1 then frame 1 and frame 2, and so on) to create a video (list of frames) with one less frame than the number of
        frames in the input GIF.

        If the output_filename is present (i.e. is not None), it should then write the resulting frames into a GIF located at the output filename string 
        using the mimwrite function from imageio, then in either case the function should return the video as a three-dimensional NumPy array.

        You may assume the file located at input_filename is in a fact a GIF (you do not need to handle the error condition where it is some other file type)

        Note: In this context a 'video' is a three-dimensional NumPy array that can best be thought of as a list of two-dimensional 'frames' which are 
        NumPy arrays.

        While debugging the last function, you can view these GIF outputs using a web browser or image viewer, they are just regular GIFs.
    '''
    video = imageio.mimread(input_filename)
    output = []
    for i in range(len(video) - 1):
        current_frame = video[i]
        next_frame = video[i+1]
        output.append((correlate_adjacent_frames(current_frame, next_frame)))

    np_output = np.array(output, dtype=np.uint8)

    if output_filename is not None:
        imageio.mimwrite(output_filename, np_output)

    return np_output

def is_triangular_path( filename ):
    ''' 
        This function takes in an input filename string of a GIF. It should use the mimread function from imageio to load the GIF into a NumPy array.
        You may assume this video is sweeping over an image in either a triangular or square path. 

        The function should then use the make_correlation_video function to get a three-dimensional NumPy array and apply some heuristic to this video in
        order to disern a triangular path video from a square path video. It should return False if the video is a square path and True if the video is a 
        triangular path. If the path is neither a triangular or square path you can do whatever (you need not handle this case).
    '''
    correlated_video = make_correlation_video(filename)

    from math import sqrt
    last_max_loc = None
    turns = 0
    for f in correlated_video:
        frame = f[0:50, 0:50]
        max_loc = (0,0)
        max_val = 0
        for r in range(0, 51, 4):
            for c in range(0, 51, 4):
                val = np.sum(frame[r:r+2,c:c+2])
                if val > max_val:
                    max_val = val
                    max_loc = (r, c)

        if last_max_loc is None:
            last_max_loc = max_loc
        else:
            change = sqrt((max_loc[0] - last_max_loc[0]) ** 2 + (max_loc[1] - last_max_loc[1])**2)
            if change > 20:
                turns += 1
            last_max_loc = max_loc

    if turns == 2:
        return True
    else:
        return False


if __name__ == "__main__":
    make_correlation_video('assets/tree-cover-square-path-0.gif', 'outputs/test_s_0.gif')
    make_correlation_video('assets/tree-cover-square-path-1.gif', 'outputs/test_s_1.gif')
    make_correlation_video('assets/tree-cover-triangle-path-0.gif', 'outputs/test_t_0.gif')
    make_correlation_video('assets/tree-cover-triangle-path-1.gif', 'outputs/test_t_1.gif')
    
