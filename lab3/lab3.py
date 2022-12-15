import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from Functions import *
from gaussfft import gaussfft
from PIL import Image, ImageFilter
from pick import pick
import random
import timeit
from mean_shift_example import mean_shift_segm
from norm_cuts_example import norm_cuts_segm
from graphcut_example import graphcut_segm
from scipy.stats import multivariate_normal


def open_image(filename):
    """
    Implement a function that opens an image file using Pillow and returns a numpy array
    with the image data. The image should be converted to floating point.

    Input arguments:
        filename - the name of the image file
    Output:
        image - the floating point image data
    """
    img = Image.open(filename)
    img = np.asarray(img).astype(np.float32)
    return img

def display_image(image, plot=plt, title=None, show=True):
    """
    Implement a function that displays an image that is in floating point format using matplotlib.

    Input arguments:
        image - the image data
    """
    img_uint8 = np.uint8(image)

    # remove axes
    plot.axis('off')

    if title is not None:
        if plot != plt:
            plot.set_title(title)
        else:
            plot.title(title)
    plot.imshow(img_uint8)
    if show:
        plot.show()

def createCentroidsFast(I, K, channels=3):
    # Use the numpy.unique function to get the unique RGB values
    values = I.reshape(-1, channels)
    # print(values.shape)
    selected_values = set()
    while len(selected_values) < K:
        # Pick a random RGB value
        print(values.shape)
        values_picked = values[np.random.choice(values.shape[0], K)]
        # Check if the value is already in the set of selected values
        for value in values_picked:
            value = tuple(value)
            if value not in selected_values:
                # Add the value to the set of selected values
                selected_values.add(value)
    return np.array(list(selected_values)[:K])


def createCentroids(I, K):
    y, x, _ = I.shape
    centroids = np.empty((K, 3), dtype = np.float32)
    for k in range(K):
        while True:
            randI = random.randint(0, y-1)
            randJ = random.randint(0, x-1)
            newValue = I[randI][randJ]
            if not np.any(centroids[:, :] == newValue):
                centroids[k] = newValue
                break
    return centroids

def kmeans_segm(image, K, L, seed = 42):
    """
    Implement a function that uses K-means to find cluster 'centers'
    and a 'segmentation' with an index per pixel indicating with
    cluster it is associated to.

    Input arguments:
        image - the RGB input image
        K - number of clusters
        L - number of iterations
        seed - random seed
    Output:
        segmentation: an integer image with cluster indices
        centers: an array with K cluster mean colors
    """
    np.random.seed(seed)
    random.seed(seed)
    # get image dimensions
    try:
        image_is_2d = True
        height, width, channels = image.shape
    except:
        image_is_2d = False
        height, channels = image.shape

    # flatten image
    if not image_is_2d:
        image_flat = np.reshape(image, (-1, channels))

    # randomly select K pixels from image
    centers = createCentroidsFast(image, K)

    # iterate L times
    for i in range(L):
        # calculate distance matrix
        dist = distance_matrix(image_flat, centers)

        # assign each pixel to closest cluster
        segmentation = np.argmin(dist, axis=1)

        # update cluster centers
        for j in range(K):
            centers[j] = np.mean(image_flat[segmentation == j], axis=0)
    
    # reshape segmentation to original image dimensions
    if image_is_2d:
        segmentation = segmentation.reshape(height, width)
    print(segmentation.shape)

    return segmentation, centers

def image_from_segmentation(segmentation, centers):
    """
    Implement a function that creates an image from a segmentation and cluster centers.

    Input arguments:
        segmentation - an integer image with cluster indices
        centers - an array with K cluster mean colors
    Output:
        image: an RGB image with cluster colors
    """ 
    # get image dimensions
    height, width = segmentation.shape

    # create image from segmentation and centers
    image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            image[i, j] = centers[segmentation[i, j]]

    return image


def mixture_prob(image, K, L, mask):
    """
    Implement a function that creates a Gaussian mixture models using the pixels 
    in an image for which mask=1 and then returns an image with probabilities for
    every pixel in the original image.

    Input arguments:
        image - the RGB input image 
        K - number of clusters
        L - number of iterations
        mask - an integer image where mask=1 indicates pixels used 
    Output:
        prob: an image with probabilities per pixel
    """ 
    # Convert PIL image to numpy array
    image = np.asarray(image)
    image = image / 255
    print(image.shape)
    #  Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).
    Ivec = np.reshape(image, (-1, 3)).astype(np.float32)
    # Reshape mask to 1D
    mask = np.reshape(mask, (-1))
    # Pick K pixels from I using mask
    Ivev_masked = Ivec[np.reshape(np.nonzero(mask == 1), (-1))]
    segmentation, µk = kmeans_segm(Ivev_masked, K, L)
    # Create covariance matrices all initialzed to 1
    cov = np.zeros((K, 3, 3))
    for i in range(K):
        cov[i] = np.eye(3) * 0.01
    # Create weights based on the amount of pixels contained in each cluster
    weights = np.zeros(K)
    for k in range(K):
        weights[k] = np.sum(np.nonzero(segmentation == k)) / segmentation.shape[0]
    # Start iterations for Expectation-Maximization
    for l in range(L):
        # Expectation
        P_ik = np.zeros((Ivec.shape[0], K))
        for k in range(K):
            # Calculate probability using multivariate normal distribution
            P_ik[:, k] = weights[k] * multivariate_normal(µk[k], cov[k]).pdf(Ivec)

        for j in range(K):
            P_ik[:, j] = np.divide(P_ik[:, j], np.sum(P_ik, axis=1), where=np.sum(P_ik, axis=1)!=0)

        # P_ik = P_ik / np.sum(P_ik, axis=1, keepdims=True)
        # Maximization
        for k in range(K):
            weights[k] = np.mean(P_ik[:, k])
            µk[k] = np.sum(P_ik[:, k].reshape(-1, 1) * Ivec, axis=0) / np.sum(P_ik[:, k])
            cov[k] = np.sum(P_ik[:, k].reshape(-1, 1, 1) * (Ivec - µk[k]).reshape(-1, 1, 3) * (Ivec - µk[k]).reshape(-1, 3, 1), axis=0) / np.sum(P_ik[:, k])

    # Compute probabilities p(c_i) in Eq.(3) for all pixels I.
    prob = np.zeros((Ivec.shape[0], K))
    for k in range(K):
        prob[:, k] = weights[k] * multivariate_normal(µk[k], cov[k]).pdf(Ivec)
        prob[:, k] = prob[:, k] / np.sum(prob[:, k])
    # prob = prob / np.sum(prob, axis=1, keepdims=True)
    prob = np.sum(prob, axis=1)
    prob = np.reshape(prob, image.shape[:2])
    return prob

def question1():
    image_path = "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/orange.jpg"
    img = open_image(image_path)
    # Show the image using matplotlib
    # display_image(img)
    # Use k-mean cluster
    segmentation, centers = kmeans_segm(img, 3, 30)

    print(f"Image shape: {img.shape}")
    print(f"Centers shape: {centers.shape}")
    print(f"Segmentation shape: {segmentation.shape}")
    # Show the segmentation using matplotlib
    display_image(image_from_segmentation(segmentation, centers), title="Segmentation")
    # Show the cluster centers using matplotlib
    display_image(np.reshape(centers, (1,centers.shape[0],centers.shape[1])), title="Cluster centers")


def question2():
    setup = """from PIL import Image;import numpy as np;from __main__ import createCentroids;from __main__ import createCentroidsFast;image_path = "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/orange.jpg";img = np.asarray(Image.open(image_path)).astype(np.float32)"""
    # Time createCentroids function against createCentroidFast using timeit
    # createCentroids
    runtime_centroid = timeit.timeit("createCentroids(img, 40)", setup=setup, number=10000)
    print(f"createCentroids time: {runtime_centroid / 10000.0}")
    # createCentroidsFast
    runtime_centroid_fast = timeit.timeit("createCentroidsFast(img, 40)", setup=setup, number=10000)
    print(f"createCentroidsFast time: {runtime_centroid_fast / 10000.0}")

    # Print the speedup factor between both function
    print(f"Speedup factor: {runtime_centroid / runtime_centroid_fast}")


def question3():
    pass

def question4():
    pass

def question5():
    # Load tiger image
    image_path = "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/tiger1.jpg"
    img = open_image(image_path)
    # Make a grid search of the parameter for the mean_shift_segm
    # function and plot the results
    spatial_bandwidth = [5, 10, 20]
    color_bandwidth = [10, 20, 30, 40, 50, 60, 70]
    num_iterations = 40
    f, ax = plt.subplots(len(spatial_bandwidth), len(color_bandwidth), figsize=(20, 10))
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, sb in enumerate(spatial_bandwidth):
        for j, cb in enumerate(color_bandwidth):
            segmentation = mean_shift_segm(img, sb, cb, num_iterations)
            Inew = overlay_bounds(img, segmentation)
            display_image(Inew, plot=ax[i][j], title=f"sb: {sb}, cb: {cb}", show=False)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    plt.show()

def question6():
    pass

def question7():
    # Load tiger image
    image_path = "/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/tiger1.jpg"
    img = Image.open(image_path)
    img = img.resize((int(img.size[0]*0.25), int(img.size[1]*0.25)))
    image_sigma = 0.5
    img = np.asarray(img.filter(ImageFilter.GaussianBlur(image_sigma))).astype(np.float32)
    # Make a grid search of the parameter for the norm_cuts_segm
    # function and plot the results

    color_bandwidth = 20
    radius = [1, 2, 3, 4, 5]
    min_area = [100, 200, 300, 400, 500]
    ncuts_thresh = 0.10
    num_iterations = 40
    max_depth = 12
    f, ax = plt.subplots(len(radius), len(min_area), figsize=(20, 10))
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, r in enumerate(radius):
        for j, ma in enumerate(min_area):
            segmentation = norm_cuts_segm(img, color_bandwidth, r, ncuts_thresh, ma, max_depth)
            Inew = overlay_bounds(img, segmentation)
            display_image(Inew, plot=ax[i][j], title=f"r: {r}, ma: {ma}", show=False)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    plt.show()

def question8():
    pass

def question9():
    pass

def question10():
    # Show the image of an orange
    img = Image.open('/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/tiger3.jpg')
    img = np.asarray(img).astype(np.float32)
    display_image(img, title="Orange")

def question11():
    scale_factor = 0.5           # image downscale factor
    area = [ 80, 110, 570, 300 ] # image region for the tiger
    # area = [249, 111, 429, 229]  # image region for the dog tiger
    K = 16                       # number of mixture components
    # Create a grid search for alpha and sigma
    alpha = [11, 14]
    sigma = [20, 25]

    img = Image.open('/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/tiger1.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))

    area = [ int(i*scale_factor) for i in area ]
    I = np.asarray(img).astype(np.float32)

    f, ax = plt.subplots(len(alpha), len(sigma), figsize=(20, 10))
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, a in enumerate(alpha):
        for j, s in enumerate(sigma):
            segm, prior = graphcut_segm(I, area, K, a, s)
            Inew = overlay_bounds(img, segm)
            display_image(Inew, plot=ax[i][j], title=f"a: {a}, s: {s}", show=False)
            ax[i][j].set_aspect('equal')
            ax[i][j].set_xticklabels([])
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    plt.show()

def question12():
    scale_factor = 0.5                  # image downscale factor
    area = [ 80, 110, 570, 300 ]        # image region for the tiger
    # area = [249, 111, 429, 229]       # image region for the dog tiger
    K = [16, 14, 12, 10, 8, 6, 4, 2]    # number of mixture components
    # Create a grid search for alpha and sigma
    alpha = 11
    sigma = 20

    img = Image.open('/Users/zach-mcc/Downloads/DD2423_Python_Labs/Images-jpg/tiger1.jpg')
    img = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)))

    area = [ int(i*scale_factor) for i in area ]
    I = np.asarray(img).astype(np.float32)

    f, ax = plt.subplots(1, len(K), figsize=(20, 10))
    f.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, k in enumerate(K):
        segm, prior = graphcut_segm(I, area, k, alpha, sigma)
        Inew = overlay_bounds(img, segm)
        display_image(Inew, plot=ax[i], title=f"K = {k}", show=False)
        ax[i].set_aspect('equal')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()

def question13():
    pass

def question14():
    pass

def question15():
    pass

funct_dict = {"Question 1": question1, "Question 2": question2, "Question 3": question3, "Question 4": question4, "Question 5": question5, "Question 6": question6, "Question 7": question7, "Question 8": question8, "Question 9": question9, "Question 10": question10,
              "Question 11": question11, "Question 12": question12, "Question 13": question13, "Question 14": question14, "Question 15": question15}


if __name__ == "__main__":
    title = 'Please choose a question to run (press SPACE to mark, ENTER to continue): '
    options = list(funct_dict.keys())
    option, index = pick(options, title)
    funct_dict[options[index]]()
