
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def plotGrid( data, title, vmin = 0, vmax = 1 ):
    """Plots a grid of images. Assumes that len(data) is a perfect square."""

    m = int(np.sqrt( len(data)) )
    f, axarr = plt.subplots(m, m)
    k = 0

    f.suptitle( title )

    for i in range(m):
        for j in range(m):

            axarr[i,j].imshow( data[k,:,:,:], vmin = 0, vmax = 255 )
            axarr[i,j].get_xaxis().set_ticks([])
            axarr[i,j].get_yaxis().set_ticks([])

            k += 1

    plt.show()

def genBoxes( scales ):

    x1 = 0.5 - 0.5 * scales
    y1 = 0.5 - 0.5 * scales
    x2 = 0.5 + 0.5 * scales
    y2 = 0.5 + 0.5 * scales

    boxes = np.array([x1, y1, x2, y2])

    return boxes.reshape( len(scales), 4 )

def scaleImages( images, scale ):
    """Scales the image relative to the centre."""

    m, h, w, c = images.shape

    #Box taken relative to the centre of the images:
    x1 = 0.5 - 0.5 * scale
    y1 = 0.5 - 0.5 * scale
    x2 = 0.5 + 0.5 * scale
    y2 = 0.5 + 0.5 * scale

    boxes = np.tile( np.array([ [x1, y1, x2, y2] ], dtype = np.float32), (m,1))
    cropSize = np.array( [h, w], dtype = np.int32)

    scaleGraph = tf.Graph()

    with scaleGraph.as_default():
        X = tf.placeholder( tf.float32, shape = (m, h, w, c) )

        scaleCrop = tf.image.crop_and_resize( X, boxes, np.array(range(m)), cropSize )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            imgagesScaled = sess.run( scaleCrop, feed_dict = { X : images } )

    return imgagesScaled

def genTranslatePrams( h, w, direction, amount ):
    """Generates parameters for translating an image"""

    a = amount
    b = 1 - a

    #Translate left:
    if direction == 0:

        offset = np.array([0.0, a], dtype = np.float32)
        size = np.array([h, np.ceil(b * w)], dtype = np.int32)
        wStart = 0
        wEnd = int( np.ceil(b * w) )
        hStart = 0
        hEnd = h

    #Translate right:
    elif direction == 1:

        offset = np.array([0.0, -a], dtype = np.float32)
        size = np.array([h, np.ceil(b * w)], dtype = np.int32)
        wStart = int(np.floor(a * w))
        wEnd = w
        hStart = 0
        hEnd = h

    #Translate up:
    elif direction == 2:

        offset = np.array([a, 0.0], dtype = np.float32)
        size = np.array([np.ceil(b * h), w], dtype = np.int32)
        wStart = 0
        wEnd = w
        hStart = 0
        hEnd = int(np.ceil(b * h))

    #Translate down:
    else:

        offset = np.array([-a, 0.0], dtype = np.float32)
        size = np.array([np.ceil(b * w), h], dtype = np.int32)
        wStart = 0
        wEnd = w
        hStart = int(np.floor(a * h))
        hEnd = h

    return offset, size, wStart, wEnd, hStart, hEnd

def translateImages( images, direction, amount ):
    """Translates the images amount in the given direction."""

    m, h, w, c = images.shape

    offset, size, wStart, wEnd, hStart, hEnd = genTranslatePrams( h, w, direction, amount )

    transGraph = tf.Graph()

    with transGraph.as_default():
        with tf.Session() as sess:

            translated = np.zeros( (m, h, w, c), dtype = np.float32 )

            sess.run(tf.global_variables_initializer())
            glimpses = tf.image.extract_glimpse( images, size, np.tile( offset, (m,1 ) ) )
            glimpses = sess.run( glimpses )

            #translated.fill(1.0) # Filling background color

            translated[:, hStart: hStart + size[0], wStart: wStart + size[1], :] = glimpses

    return translated

def rotateK90Degs( images, k ):
    """Rotates the images by k*90 degrees, counter clockwise."""

    m, h, w, c = images.shape

    rot90Graph = tf.Graph()

    with rot90Graph.as_default():

        X    = tf.placeholder(tf.float32, shape = (m, h, w, c) )
        rots = tf.placeholder(tf.int32)
        rotate = tf.image.rot90(X, k = rots )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            rotated = sess.run( rotate, feed_dict = { X : images, rots : k})

    return rotated

def rotateImages( images, angle ):
    """Rotates the images by angle counter clockwise."""

    m, h, w, c = images.shape

    rotGraph = tf.Graph()

    with rotGraph.as_default():

        X = tf.placeholder( tf.float32, shape = (m, h, w, c) )
        rads = tf.placeholder( tf.float32, shape = ( m ) )
        rotateImages = tf.contrib.image.rotate( X, rads )

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            rotated = sess.run( rotateImages, feed_dict = { X: images, rads : np.repeat( angle, m) } )

    return rotated

def mirrorImages( images, axis ):
    """Mirrors the images horizontally (axis == 0) or vertically (axis == 1)."""

    m, h, w, c = images.shape

    mirrorGraph = tf.Graph()

    with mirrorGraph.as_default():
        X = tf.placeholder( tf.float32, shape = [m, h, w, c] )

        if ( axis == 0 ):
            #mirrorFlip = tf.image.flip_left_right( X )
            mirrorFlip = tf.map_fn( tf.image.flip_left_right, X )

        else:
            #mirrorFlip = tf.image.flip_up_down( X )
            mirrorFlip = tf.map_fn( tf.image.flip_up_down, X )

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            mirrored = sess.run( mirrorFlip, feed_dict = { X : images })

    return mirrored

def adjustBrightness( images, delta ):
    """Adjusts the rightness of the images by delta."""

    m, h, w, c = images.shape

    brightGraph = tf.Graph()

    with brightGraph.as_default():
        X = tf.placeholder( tf.float32, shape = (m, h, w, c) )
        d = tf.placeholder( tf.float32 )

        bright = tf.image.adjust_brightness( X, d )
        newImages = tf.clip_by_value( bright, 0.0, 255.0 )

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            brightened = sess.run( newImages, feed_dict = { X : images, d : delta } )

    return brightened

