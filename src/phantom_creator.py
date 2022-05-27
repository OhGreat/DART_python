import numpy as np
from PIL import Image
from os import makedirs
from os.path import exists

def create_image(x,y):
    """ Basic function required to create phantoms
    """
    image = np.ones([x,y])
    # Resize Image
    diag = len(np.diag(image)//2)
    if x == 80 and y == 80:
        plus = 8
    else: 
        plus = 10
    image = np.pad(image, pad_width=diag+plus)
    
    _ = np.linspace(-1, 1, image.shape[0])
    xv, yv = np.meshgrid(_,_)
    return image, xv, yv

def create_semilunars(img_size=512, gray_values=[255, 200, 150], n=1, 
                        overlap=False, seed=None, img_name=None):
    """ Create lunar like phantoms.

        Input:
            - img_size: defines the size of the image,
                should be 256 or 512.
            - gray_values: list with three values representing the 
                gray values to use for the images.
            - n: defines number of images to generate.
            - overlap: defines wether to make circles overlap or not
            - seed: defines the seed in order to have reproducable phantoms
            - img_name: string defining the path and filename to use.
                The filename should not have the extension, 
                it will be created as a png by default.
                Example of definition of img_name:
                    img_name = "dir/to/save/filename"

        Output:
            returns a list of phantoms. (Phantoms as numpy arrays)
    """

    if img_size == 512:
        x, y = 164, 164
    elif img_size == 256:
        x, y = 80, 80
    else:
        exit("img_size can only be set to 512 or 256")

    if seed != None:
        np.random.seed(seed)

    # define save paths
    if img_name != None:
        if not exists('data/'):
            makedirs('data/')
    
    circles = []
    for i in range(n):
        
        image, xv, yv = create_image(x,y)
        mu, sigma = 0 , 0.1
        noise =  np.abs(np.random.normal(mu, sigma, 2))
        
        image[xv**2 + yv**2 >0.6] = gray_values[0]
        image[xv**2 + yv**2 <0.49] = gray_values[2]
        if overlap == True:
            image[(xv-0.1+noise[0])**2 + (yv-0.1+noise[0])**2 <0.3] = gray_values[0]
            image[(xv-0.1+noise[1])**2 + (yv-0.1+noise[1])**2 <0.19] = gray_values[2]
            image[(xv-0.1+noise[0])**2 + (yv-0.1+noise[0])**2 <0.11] = gray_values[1]
            image[(xv-0.2+noise[1])**2 + (yv-0.2+noise[1])**2 <0.02] = gray_values[2]
        else: 
            image[(xv-0.1+noise[0])**2 + (yv-0.1+noise[0])**2 <0.3] = gray_values[0]
            image[(xv-0.1+noise[0])**2 + (yv-0.1+noise[0])**2 <0.19] = gray_values[2]
            image[(xv-0.1+noise[0])**2 + (yv-0.1+noise[0])**2 <0.11] = gray_values[1]
            image[(xv-0.2+noise[0])**2 + (yv-0.2+noise[0])**2 <0.02] = gray_values[2]
            
        circles.append(image.astype(np.uint8))
        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png")
        
    return circles