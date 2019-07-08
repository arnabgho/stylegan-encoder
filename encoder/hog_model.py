import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16 #, preprocess_input
from keras.preprocessing import image
import keras.backend as K
from skimage import draw
import cv2
import matplotlib.pyplot as plt

################# HOG Helper Functions #####################

def get_sobel_kernel(ksize):
    if (ksize % 2 == 0) or (ksize < 1):
        raise ValueError("Kernel size must be a positive odd number")
    _base = np.arange(ksize) - ksize//2
    a = np.broadcast_to(_base, (ksize,ksize))
    b = ksize//2 - np.abs(a).T
    s = np.sign(a)
    return a + s*b


def get_gaussian_kernel(ksize = 3, sigma = -1.0):
    ksigma = 0.15*ksize + 0.35 if sigma <= 0 else sigma
    i, j   = np.mgrid[0:ksize,0:ksize] - (ksize-1)//2
    kernel = np.exp(-(i**2 + j**2) / (2*ksigma**2))
    return kernel / kernel.sum()


def get_laplacian_of_gaussian_kernel(ksize = 3, sigma = -1.0):
    ksigma = 0.15*ksize + 0.35 if sigma <= 0 else sigma
    i, j   = np.mgrid[0:ksize,0:ksize] - (ksize-1)//2
    kernel = (i**2 + j**2 - 2*ksigma**2) / (ksigma**4) * np.exp(-(i**2 + j**2) / (2*ksigma**2))
    return kernel - kernel.mean()


def tf_kernel_prep_4d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1, 1)).swapaxes(0,2).swapaxes(1,3)


def tf_kernel_prep_3d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0,1).swapaxes(1,2)


def tf_filter2d(batch, kernel, strides=(1,1), padding='SAME'):
    n_ch = batch.shape[3].value
    tf_kernel = tf.constant(tf_kernel_prep_4d(kernel, n_ch))
    return tf.nn.depthwise_conv2d(batch, tf_kernel, [1, strides[0], strides[1], 1], padding=padding)


def tf_deriv(batch, ksize=3, padding='SAME'):
    try:
        n_ch = batch.shape[3].value
    except:
        n_ch = batch.shape[3]
        #n_ch = int(batch.get_shape()[3])
    gx = tf_kernel_prep_3d(np.array([[ 0, 0, 0],
                                     [-1, 0, 1],
                                     [ 0, 0, 0]]), n_ch)
    gy = tf_kernel_prep_3d(np.array([[ 0, -1, 0],
                                     [ 0, 0, 0],
                                     [ 0, 1, 0]]), n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel", dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")


def tf_sobel(batch, ksize=3, padding='SAME'):
    n_ch = batch.shape[3] #batch.shape[3].value
    gx = tf_kernel_prep_3d(get_sobel_kernel(ksize),   n_ch)
    gy = tf_kernel_prep_3d(get_sobel_kernel(ksize).T, n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


def tf_sharr(batch, ksize=3, padding='SAME'):
    n_ch = batch.shape[3] #batch.shape[3].value
    gx = tf_kernel_prep_3d([[ -3, 0,  3],
                            [-10, 0, 10],
                            [ -3, 0,  3]], n_ch)
    gy = tf_kernel_prep_3d([[-3,-10,-3],
                            [ 0,  0, 0],
                            [ 3, 10, 3]], n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


def tf_laplacian(batch, padding='SAME'):
    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]], dtype=batch.dtype)
    return tf_filter2d(batch, kernel, padding=padding)


def tf_boxfilter(batch, ksize = 3, padding='SAME'):
    kernel = np.ones((ksize, ksize), dtype=batch.dtype) / ksize**2
    return tf_filter2d(batch, kernel, padding=padding)

def tf_rad2deg(rad):
    return 180 * rad / tf.constant(np.pi)

######### Draw and Save HOG Function ##############

def hog_image(img, hist, cell_size=8, orientations=9):
    c_row, c_col = (cell_size, cell_size)
    s_row, s_col = img.shape[:2]

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    radius = min(c_row, c_col) // 2 - 1
    orientations_arr = np.arange(orientations)
    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
        np.pi * (orientations_arr + .5) / orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((s_row, s_col), dtype=float)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2,
                                c * c_col + c_col // 2])
                rr, cc = draw.line(int(centre[0] - dc),
                                   int(centre[1] + dr),
                                   int(centre[0] + dc),
                                   int(centre[1] - dr))
                hog_image[rr, cc] += hist[r, c, o]

    return hog_image

######### HOG Function ############################

def tf_select_by_idx(a, idx, grayscale):
    if grayscale:
        return a[:,:,:,0]
    else:
        return tf.where(tf.equal(idx, 2),
                         a[:,:,:,2],
                         tf.where(tf.equal(idx, 1),
                                   a[:,:,:,1],
                                   a[:,:,:,0]))


def tf_hog_descriptor(images, cell_size = 8, block_size = 2, block_stride = 1, n_bins = 9,
                      grayscale = False):

    batch_size, height, width, depth = images.shape
    scale_factor = tf.constant(180 / n_bins, name="scale_factor", dtype=tf.float32)

    img = images #tf.constant(images, name="ImgBatch", dtype=tf.float32)

    if grayscale:
        img = tf.image.rgb_to_grayscale(img, name="ImgGray")


    # automatically padding height and width to valid size (multiples of cell size)
    if height % cell_size != 0 or width % cell_size != 0:
        height = height + (cell_size - (height % cell_size)) % cell_size
        width = width + (cell_size - (width % cell_size)) % cell_size
        img = tf.image.resize_image_with_crop_or_pad(img, height, width)


    # gradients
    grad = tf_deriv(img)
    g_x = grad[:,:,:,0::2]
    g_y = grad[:,:,:,1::2]

    # masking unwanted gradients of edge pixels
    mask_depth = 1 if grayscale else depth
    g_x_mask = np.ones((batch_size, height, width, mask_depth))
    g_y_mask = np.ones((batch_size, height, width, mask_depth))
    g_x_mask[:, :, (0, -1)] = 0
    g_y_mask[:, (0, -1)] = 0
    g_x_mask = tf.constant(g_x_mask, dtype=tf.float32)
    g_y_mask = tf.constant(g_y_mask, dtype=tf.float32)

    g_x = g_x*g_x_mask
    g_y = g_y*g_y_mask

    # maximum norm gradient selection
    g_norm = tf.sqrt(tf.square(g_x) + tf.square(g_y), "GradNorm")

    if not grayscale and depth != 1:
        # maximum norm gradient selection
        idx    = tf.argmax(g_norm, 3)
        g_norm = tf.expand_dims(tf_select_by_idx(g_norm, idx, grayscale), -1)
        g_x    = tf.expand_dims(tf_select_by_idx(g_x,    idx, grayscale), -1)
        g_y    = tf.expand_dims(tf_select_by_idx(g_y,    idx, grayscale), -1)

    g_dir = tf_rad2deg(tf.atan2(g_y, g_x)) % 180
    g_bin = tf.to_int32(g_dir / scale_factor, name="Bins")

    # cells partitioning
    cell_norm = tf.space_to_depth(g_norm, cell_size, name="GradCells")
    cell_bins = tf.space_to_depth(g_bin,  cell_size, name="BinsCells")

    # cells histograms
    hist = list()
    zero = tf.zeros(cell_bins.get_shape())
    for i in range(n_bins):
        mask = tf.equal(cell_bins, tf.constant(i, name="%i"%i))
        hist.append(tf.reduce_mean(tf.where(mask, cell_norm, zero), 3))
    hist = tf.transpose(tf.stack(hist), [1,2,3,0], name="Hist")
    # blocks partitioning
    block_hist = tf.extract_image_patches(hist,
                                          ksizes  = [1, block_size, block_size, 1],
                                          strides = [1, block_stride, block_stride, 1],
                                          rates   = [1, 1, 1, 1],
                                          padding = 'VALID',
                                          name    = "BlockHist")

    # block normalization
    block_hist = tf.nn.l2_normalize(block_hist, 3, epsilon=1.0)

    # HOG descriptor
    hog_descriptor = tf.reshape(block_hist,
                                [int(block_hist.get_shape()[0]),
                                 int(block_hist.get_shape()[1]) * \
                                 int(block_hist.get_shape()[2]) * \
                                 int(block_hist.get_shape()[3])],
                                 name='HOGDescriptor')
    return hog_descriptor, block_hist, hist

################ Perceptual Model Like Function #####################

def preprocess_input(img):
    return img/255.0

def display_hog(img_tensor,h):
    #h = h.eval()
    tf_hog_image = hog_image(img_tensor[0], h[0])
    plt.figure(figsize=(8,8))
    plt.imshow(tf_hog_image, cmap=plt.cm.gray)
    plt.show()
    plt.savefig('hog.png')





def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img,(img_size,img_size))
        #img = img.reshape([-1,img_size,img_size,3])  #image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    loaded_images = np.float32(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images

class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def build_perceptual_model(self, generated_image_tensor):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        #self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,                           ### might have to change the preprocessing function
                                                                  (self.img_size, self.img_size), method=1))
        #generated_img_features , _ , h = tf_hog_descriptor(generated_image,grayscale=True)
        #generated_img_features = tf_sobel(generated_image)
        generated_img_features = tf_sharr(generated_image)

        #h = h.eval()
        #display_hog(generated_image,h)

        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.sess.run([self.features_weight.initializer, self.features_weight.initializer])

        self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
                                                 self.features_weight * generated_img_features) #/ 828.900# / (82890.0) : weight for previous occasion, might have to change

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        #image_features, _ , h = tf_hog_descriptor(loaded_image,grayscale=True) #self.perceptual_model.predict_on_batch(loaded_image)

        #image_features = tf_sobel(loaded_image)
        image_features = tf_sharr(loaded_image)

        #h = h.eval()
        #display_hog(loaded_image,h)
        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

            print("Entered Complicated if statement")

        self.sess.run(tf.assign(self.features_weight, weight_mask))
        self.sess.run(tf.assign(self.ref_img_features, image_features))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #self.sess.run(tf.global_variables_initializer())
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss
            #input()

