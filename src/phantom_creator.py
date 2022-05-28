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
        
        image[xv**2 + yv**2 >0.6] = 0
        image[xv**2 + yv**2 <0.49] = 170
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




def create_alien(img_size=512, gray_values=[40,80,100], n=1, 
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
    
    aliens = []
    for i in range(n):
        
        image, xv, yv = create_image(x,y)
        mu, sigma = 0 , 0.05
        noise =  np.abs(np.random.normal(mu, sigma, 2))
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
            
            
        aliens.append(image)
        
  

        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png")
        
    return aliens

    

def create_paws(img_size=512, gray_values=[50,110,120], n=1,
                                 seed=None, img_name=None):
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
    
    paws = []
    for i in range(n):

        image, xv, yv = create_image(x,y)
        thresh = np.random.choice(range(11),3)
        shift_x = np.random.choice(range(50))
        shift_y = np.random.choice(range(3))

        mu, sigma = 0 , 0.1
        noise =  np.abs(np.random.normal(mu, sigma, 2))

        image[xv**2 + yv**2 >0.1] = 0
        image[xv**2 + yv**2 <0.09] = 110

        if img_size==256:
            image[25:231,30:40] = 120
            image[25:231, 216:226] = 120
            image[25:35, 40:216] = 120
            image[221:231, 40:216] = 120
            image[180:210+shift_y,50:80+shift_x] = 120
        else: 
            image[50:462,60:80] = 120
            image[50:462, 432:452] = 120
            image[50:70, 80:432] = 120
            image[442:462, 80:432] = 120
            image[360:420+shift_y,100:160+shift_x] = 120

        image[(xv-0.5)**2/0.2+(yv+0.3-noise[0])**2<0.05] = 110

        if thresh[0] > 5: 
            image[(xv-0.006)**2/0.4+(yv+0.18)**2/0.7<0.01] = gray_values[0]
        if thresh[1] <4: 
            image[(xv+0.17)**2/0.4+(yv+0.1)**2/0.7<0.01] = gray_values[0]
         
        image[(xv-0.17)**2/0.4+(yv+0.1)**2/0.7<0.01] = gray_values[0]
        image[(xv-0.01)**2/0.7+(yv-0.1)**2/0.7<0.02] = gray_values[0]
        
        paws.append(image)
  

        if img_name != None:
            Image.fromarray(image.astype(np.uint8)).save(f"{img_name}_{i}.png")
        
    return paws