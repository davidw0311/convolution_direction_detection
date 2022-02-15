from sys import argv

from implementation import make_correlation_video

if __name__ == '__main__':
    helpstring = 'usage: python make_correlation_video.py <input filename> <output filename>' 
    argc       = len( argv )

    if argc < 3 or argc > 3:
        print( helpstring )
    else:
        make_correlation_video( argv[1], argv[2] )
