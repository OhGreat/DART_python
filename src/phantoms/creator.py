import numpy as np
from PIL import Image
from os import makedirs
from os.path import exists

def create_phantoms(phantoms="semilunars",img_size=512, gray_values=[80,120,180], n=1, 
                        overlap=False, seed=None, img_name=None):
    generated = []
    if phantoms == "semilunars":
        generated = create_semilunars(img_size=img_size, gray_values=gray_values,
                                    n=n, overlap=overlap, seed=seed, img_name=img_name)
    elif phantoms == "aliens":
        generated = create_aliens(img_size=img_size, gray_values=gray_values,
                                    n=n, overlap=overlap, seed=seed, img_name=img_name)
    elif phantoms == "clouds":
        generated = create_clouds(img_size=img_size, gray_values=gray_values,
                                    n=n, overlap=overlap, seed=seed, img_name=img_name)
    elif phantoms == "paws":
        generated = create_paws(img_size=img_size, gray_values=gray_values,
                                    n=n, overlap=overlap, seed=seed, img_name=img_name)
    else:
        exit("please choose a valid class.")
    return generated

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

def create_semilunars(img_size=512, gray_values=[80,120,180], n=1, 
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
    # control correct image size
    if img_size == 512:
        x, y = 164, 164
    elif img_size == 256:
        x, y = 80, 80
    else:
        exit("img_size can only be set to 512 or 256")
    # control seed
    if seed != None:
        np.random.seed(seed)
    # define save paths
    if img_name != None:
        if not exists('data/'):
            makedirs('data/')
    # create phantoms
    semilunars = []
    for i in range(n):
        image, xv, yv = create_image(x,y)
        # define noise for randomization
        mu, sigma = 0 , 0.1
        noise =  np.abs(np.random.normal(mu, sigma, 2))
        # define shapes in image
        image[xv**2 + yv**2 >0.6] = 0
        image[xv**2 + yv**2 <0.49] = gray_values[1]
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
        semilunars.append(image.astype(np.uint8))
        # save image
        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png")
    return semilunars




def create_aliens(img_size=512, gray_values=[40,80,100], n=1, 
                overlap=False, seed=None, img_name=None):
    """ Create alien like phantoms.

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
    # control image size
    if img_size == 512:
        x, y = 164, 164
    elif img_size == 256:
        x, y = 80, 80
    else:
        exit("img_size can only be set to 512 or 256")
    # seed control
    if seed != None:
        np.random.seed(seed)
    # define save paths
    if img_name != None:
        if not exists('data/'):
            makedirs('data/')
    # create alien phantoms
    aliens = []
    for i in range(n):
        image, xv, yv = create_image(x,y)
        # create noise for randomization
        mu, sigma = 0 , 0.05
        noise =  np.abs(np.random.normal(mu, sigma, 2))
        # create phantom
        image[(xv-0.01)**2/0.5+(yv+0.01)**2>0.5] = 0
        image[(xv-0.01)**2/0.5+(yv+0.01)**2<0.48] = gray_values[2]
        if overlap == True:
            image[(xv+0.3-noise[0])**2/0.2+(yv+0.01-noise[0])**2<0.05] = gray_values[1]
            image[(xv+0.3-noise[1])**2/0.2+(yv+0.1-noise[1])**2<0.01] = gray_values[0]
            image[(xv-0.3+noise[0])**2/0.2+(yv+0.01+noise[0])**2<0.05] = gray_values[1]
            image[(xv-0.3+noise[1])**2/0.2+(yv+0.1+noise[1])**2<0.01] = gray_values[0]
            image[(xv-0.01-noise[0])**2+(yv-0.6+noise[0])**2/0.025<0.02] = gray_values[0]
        else: 
            image[(xv+0.3-noise[0])**2/0.2+(yv+0.01-noise[0])**2<0.05] = gray_values[1]
            image[(xv+0.3-noise[0])**2/0.2+(yv+0.1-noise[0])**2<0.01] = gray_values[0]
            image[(xv-0.3+noise[0])**2/0.2+(yv+0.01-noise[0])**2<0.05] = gray_values[1]
            image[(xv-0.3+noise[0])**2/0.2+(yv+0.1-noise[0])**2<0.01] = gray_values[0]
            image[(xv-0.01-noise[0])**2+(yv-0.6+noise[0])**2/0.025<0.02] = gray_values[0] 
        aliens.append(image.astype(np.uint8))
        # save image
        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png") 
    return aliens

def create_paws(img_size=512, gray_values=[80,120,180], n=1, 
                        overlap=False, seed=None, img_name=None):
    """ Create paw like phantoms.

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
    # img size control
    if img_size == 512:
        x, y = 164, 164
    elif img_size == 256:
        x, y = 80, 80
    else:
        exit("img_size can only be set to 512 or 256")
    # control seed
    if seed != None:
        np.random.seed(seed)
    # define save paths
    if img_name != None:
        if not exists('data/'):
            makedirs('data/')
    paws = []
    for i in range(n):
        image, xv, yv = create_image(x,y)

        thresh = np.random.choice(range(20),3)
        shift_x = np.random.choice(range(20,100))
        shift_y = np.random.choice(range(3))

        mu, sigma = 0 , 0.5
        noise =  np.abs(np.random.normal(mu, sigma, 2))

        # create images
        image[xv**2 + yv**2 >0.1] = 0
        image[xv**2 + yv**2 <0.09] = 255
        if img_size==256:
            image[25:231,30:40] = 255
            image[25:231, 216:226] = 255
            image[25:35, 40:216] = 255
            image[221:231, 40:216] = 255
            image[180:210+shift_y,50:80+shift_x] = 255
        else: 
            image[50:462,60:80] = 255
            image[50:462, 432:452] = 255
            image[50:70, 80:432] = 255
            image[442:462, 80:432] = 255
            image[360:420+shift_y,100:160+shift_x] = 255

        image[(xv-0.5)**2/0.2+(yv+0.3-noise[0])**2<0.05] = 255

        if thresh[0] <7: 
            image[(xv-0.006)**2/0.4+(yv+0.18)**2/0.7<0.01] = 0
        if thresh[1] > 5 : 
            image[(xv+0.17)**2/0.4+(yv+0.1)**2/0.7<0.01] = 0
        
        image[(xv-0.17)**2/0.4+(yv+0.1)**2/0.7<0.01] = 0
        image[(xv-0.01)**2/0.7+(yv-0.1)**2/0.7<0.02] = 0

        #fix the fucking pixels == 1
        image[image == 1] = 0
        
        paws.append(image.astype(np.uint8))
        # save image
        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png") 
    return paws


def create_clouds(img_size=512, gray_values=[80,120,180], n=1, 
                        overlap=False, seed=None, img_name=None):
    """ Create cloud like phantoms.

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
    # img size control
    if img_size == 512:
        x, y = 164, 164
    elif img_size == 256:
        x, y = 80, 80
    else:
        exit("img_size can only be set to 512 or 256")
    # control seed
    if seed != None:
        np.random.seed(seed)
    # define save paths
    if img_name != None:
        if not exists('data/'):
            makedirs('data/')


    clouds=[]
    for i in range(n):
        image, xv, yv = create_image(x,y)

        noise = np.random.normal(0,0.3,3)

        image[(xv-0.1)**2+(yv+0.01)**2/0.4<0.6] = 255
        image[(xv-0.1)**2+(yv+0.5)**2/0.4<0.1] = 255
        image[(xv-0.1)**2+(yv-0.5)**2/0.4<(0.1+noise[1])] = 255
        image[(xv+0.4)**2+(yv+0.5)**2/0.4<0.1] = 255
        image[(xv-0.4)**2+(yv-0.5)**2/0.4<(0.1+noise[1])] = 255
        image[(xv-0.6)**2+(yv+0.3)**2/0.4<(0.1-noise[2])] = 255
        image[(xv+0.4)**2+(yv-0.3)**2/0.4<(0.1-noise[2])] = 255
        
        n = np.random.normal(0,0.2,1)

        if 0.15<n[0]>0.3 :
            image[(xv-0.1)**2+(yv-0.1)**2/0.4<0.02] = 0
            image[(xv+0.4)**2+(yv-0.2)**2/0.4<0.005] = 0
            image[(xv-0.1)**2+(yv+0.2)**2/0.4<0.01] = 0
            image[(xv-0.4)**2+(yv+0.2)**2/0.4<0.02] = 0
            image[(xv+0.3)**2+(yv+0.3)**2/0.4<0.01] = 0
            image[(xv-0.4)**2+(yv-0.3)**2/0.4<0.02] = 0
        elif n[0]<0.09:
            image[(xv+0.4)**2+(yv-0.2)**2/0.4<0.005] = 0
            image[(xv-0.4)**2+(yv+0.2)**2/0.4<0.02] = 0
            image[(xv-0.4)**2+(yv-0.3)**2/0.4<0.02] = 0
            image[(xv+0.3)**2+(yv+0.3)**2/0.4<0.01] = 0
        else: 
            image[(xv+0.3)**2+(yv+0.3)**2/0.4<0.01] = 0
            image[(xv-0.4)**2+(yv-0.3)**2/0.4<0.02] = 0
            image[(xv-0.1)**2+(yv-0.1)**2/0.4<0.02] = 0

        clouds.append(image.astype(np.uint8))

        # save image
        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png") 
    return clouds