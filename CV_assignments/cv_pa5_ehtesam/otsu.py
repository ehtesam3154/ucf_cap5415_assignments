import glob
import warnings
from os.path import join
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

#manuel thresholding with numpy histogram
def plot_hist_binarization(image, hist, threshold_list):
    '''
    Plot three binarized images based on the three given thresholds
    '''
    # Create Subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title(f"Original Image", y=1.02)

    # Plot histogram
    ax[1].plot(np.arange(0, 256), hist, color='tab:blue', label='histogram')
    ax[1].legend()
    ax[1].set_title(f"Histogram", y=1.02)

    plt.tight_layout()
    plt.show()

    # Create Subplots
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    # Set Figure facecolor to White
    fig.set_facecolor('white')

    # Plot first threshold
    plt.subplot(231)
    plt.plot(np.arange(0, 256), hist,
             color='tab:blue',
             label='histogram')
    plt.axvline(x=threshold_list[0],
                color='tab:red',
                linestyle='dashed',
                label='threshold')
    plt.legend()
    plt.title(f"Threshold {threshold_list[0]}", y=1.02)

    # Plot binalized image
    plt.subplot(234)
    plt.imshow(image > threshold_list[0], cmap='gray')
    plt.axis('off')

    # Plot second threshold
    plt.subplot(232)
    plt.plot(np.arange(0, 256), hist,
             color='tab:blue',
             label='histogram')
    plt.axvline(x=threshold_list[1],
                color='tab:red',
                linestyle='dashed',
                label='threshold')
    plt.legend()
    plt.title(f"Threshold {threshold_list[1]}", y=1.02)

    # Plot binalized image
    plt.subplot(235)
    plt.imshow(image > threshold_list[1], cmap='gray')
    plt.axis('off')

    # Plot second threshold
    plt.subplot(233)
    plt.plot(np.arange(0, 256), hist,
             color='tab:blue',
             label='histogram')
    plt.axvline(x=threshold_list[2],
                color='tab:red',
                linestyle='dashed',
                label='threshold')
    plt.legend()
    plt.title(f"Threshold {threshold_list[2]}", y=1.02)

    # Plot binalized image
    plt.subplot(236)
    plt.imshow(image > threshold_list[2], cmap='gray')
    plt.axis('off')

    plt.show()

def process_image(image_path, thresholds):
    '''
    Process an image by plotting binarized images for given thresholds
    '''
    # Load image in grayscale
    image = cv.imread(image_path, 0)
    # Calculate histogram using NumPy
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    # Plot thresholds
    plot_hist_binarization(image, hist, thresholds)

#otsu thresholding functions

def threshold_image(image, threshold):
    '''
    Binarizes the image based on the given threshold
    '''
    # Initialize an array to store the thresholded image
    thresholded_image = np.zeros(image.shape)
    
    # Set pixels above the threshold to 1
    thresholded_image[image >= threshold] = 1
    
    return thresholded_image

def compute_otsu_criteria(image, threshold):
    '''
    Computes the Otsu criteria for a given threshold
    '''
    # Binarize the image based on the threshold
    thresholded_image = threshold_image(image, threshold)
    
    # Calculate the total number of pixels
    nb_pixels = image.size
    
    # Count the number of pixels in the two classes
    nb_pixels1 = np.count_nonzero(thresholded_image)
    
    # Calculate weights for the two classes
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1
    
    # Check for division by zero
    if weight1 == 0 or weight0 == 0:
        return np.inf
    
    # Get pixel values for each class
    val_pixels1 = image[thresholded_image == 1]
    val_pixels0 = image[thresholded_image == 0]
    
    # Calculate variances for each class
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    
    # Compute the Otsu criteria
    otsu_criteria = weight0 * var0 + weight1 * var1
    
    return otsu_criteria

def find_best_threshold(image):
    '''
    Finds the best Otsu threshold for the given image
    '''
    # Define the range of possible thresholds
    threshold_range = range(np.max(image) + 1)
    
    # Compute the Otsu criteria for each threshold
    criterias = [compute_otsu_criteria(image, threshold) for threshold in threshold_range]
    
    # Find the threshold that minimizes the Otsu criteria
    best_threshold = threshold_range[np.argmin(criterias)]
    
    return best_threshold


def plot_otsu_binarization(image):
    '''
    Plots binarized images based on Otsu thresholding
    '''
    # Create Subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor='white')

    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Original Image", y=1.02)

    # Plot histogram
    axes[1].plot(np.arange(256), np.histogram(image.flatten(), 
                                              bins=256, range=[0, 256])[0],
                                              color='tab:blue', label='histogram')
    axes[1].set_title("Histogram", y=1.02)
    axes[1].axvline(x=find_best_threshold(image), 
                    color='tab:red', linestyle='dashed',
                    label='threshold')
    axes[1].legend()

    # Plot Otsu thresholding
    axes[2].imshow(image > find_best_threshold(image), cmap='gray')
    axes[2].axis('off')
    axes[2].set_title("Otsu Thresholding", y=1.02)

    plt.show()

# Get image list
image_list = glob.glob(join('images', '*.jpg'))
print(image_list)

# manual thresholding
# Process each image with specific thresholds
process_image(image_list[0], [70, 110, 140])
process_image(image_list[1], [60, 100, 130])
process_image(image_list[2], [25, 70, 120])



#otsu thresholding
for i in range(3):
    # Load image in grayscale
    image = cv.imread(image_list[i], 0)
    # Plot three thresholds
    plot_otsu_binarization(image)