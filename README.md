# Discrete Algebric Reconstruction Technique (DART) - *currently in development*
DART is an iterative reconstruction algorithm for discrete tomography. The original publication in <a href="#original_publication">[1]</a> was used as reference to create this library.
What this repository consists of, is an implementation of the DART algorithm together with a framework to generate phantoms and measurements, to test the algorithm itself.

## DART

### ART reconstruction step
The DART algorithm, implemented as in the original publication <a href="#original_publication"> [1]</a>, alternates between continuous and discrete reconstruction steps. For the continuous step, many reconstruction algorithms were implemented with **astra-toolbox**. Publications relevant to this library can be found in <a href="#astra_1">[2]</a>, <a href="#astra_2">[3]</a> and <a href="#astra_3">[4]</a>. For the original publication of SART, which is the main reconstruction algorithm presented in the original DART publication, you can refer to <a href="#SART">[5]</a>.

## Prerequisites

`Python 3` and the following packages are required to use this library:

- `numpy`
- `Pillow`
- `astra-toolbox` : required to create phantoms and projections, documentation is available <a href="https://www.astra-toolbox.com/">here</a>.

## Usage
To run DART, data *(in the form of phantoms)* and measurements *(in the form of projections and detector values)* need to be artificially constructed. 
Therefore, three main wrappers have been created around the code to solve the following tasks:
- phantom creation
- projection and measurements acquisition
- running DART reconstruction algorithm

Usage of the framework for each of this tasks is described in detail in the following sections.

### Generating phantoms

#### Semilunar phantoms

To generate semilunar like phantoms you can use the `create_semilunar` function. It can be imported from `phantom_creator.py` and used as below:
```
from phantom_creator import create_semilunars
phantom_list = create_semilunars(img_size=512, gray_values=[255, 200, 150], n=3, overlap=False, seed=None, img_name="dir/to/save/filename")
``` 
Parameters:
- `img_size`: should be an integer of value 256 or 512 and represents the size of the phantoms to generate.
- `gray_values`: list of three integers between 0 and 255 representing the intensities to use for the various phantom layers.
- `n`: integer representing the number of phantoms to generate.
- `overlap`: defines wether the shape layers should overlap between them.
- `seed`: integer to be passed to have reproducible results.
- `img_name`: string defining the path and filename to use.
                The filename should not have the extension, 
                it will be created as a png by default.

Output:
- python list of phantoms. (Phantoms as numpy arrays)

### Generating projections

#### From 2D phantoms
To generate measurements in the form of 1D projections, the function `project_from_2D` has been created. You can import it and use it with the following commands:

```python
from measurements.projections import project_from_2D
proj_id, sino_id, sinogram = project_from_2D(phantom_id, vol_geom, n_projections, detector_spacing, apply_noise=False, save_dir=None, use_gpu=False)
```
Parameters:
- `phantom_id`: Phantom as astra-toolbox object.
- `vol_geom`: geometry of the output image. Used to define the number of detectors as the first dimension of the vol_geom.
- `n_projections`: is an integer value representing the number of projections as the number of angles to make measurements from.
- `detector_spacing`: defines the size of the pixel.
- `apply_noise`: boolean value that adds Poisson distributed noise to the image when set to True. False by default.
- `save_dir`: string representing the directory to save png images that represent the measurements. Images won't be saved if this parameter is not set.
- `use_gpu`: creates a projector that can use GPU  

Output:
- The function will return `proj_id`, `sino_id` and `sinogram`. The first is a reference to the astra toolkit projector object, the second is a reference to the astra toolkit sinogram object and the former is the sinograms' actual measurements.

#### From 3D phantoms

### Running DART
All the steps required to run the DART algorithm have been broken down and can be used separately. A detailed desctiption for the usage of all the functions available in the library will follow in this section.
 
#### DART Instance
DART can be imported and initialized in the following way:
```python
from reconstruction_algs.DART import DART
dart_instance = DART()
```
### Segmentation
The method `segment` can be used to segment an image, given the range of gray values:
```python
segmented_img = dart.segment(img, gray_levels)
```
Parameters:
- `img`: is the grayscale input phantom to segment as a 2D numpy matrix.
- `gray_levels` : array of gray levels to compute the thresholds for the segmentation from.

Output:
- returns the segmented image as a numpy array

### Pixel neighborhood
To calculate the indexes of neighbours of a specific pixel, you can use the method `pixel_neighborhood` as below:
```python
neighbours = dart.pixel_neighborhood(img, x, y)
```
Parameters:
- `img` : is as usual the phantom in the form of 2D numpy matrix.
- `x,y`: are the coordinates of the pixel in 

Output:
- The method returns a 2D array containing arrays of the x,y coordinates of the neighbours.

### Boundary pixels
To calculate the boundary pixels of the phantom image, the method `boundary_pixels` takes as input the phantom *as a numpy array*, and calculates the boundary pixels with the help of the `pixel_neighborhood` method described above. You can use it as follows:
```python
b_pixels = dart.boundary_pixels(img):
```
Parameters:
- `b_pixels`: is the output given as a 2D binary mask of the image, where True values represent that the given pixel is a boundary pixel.

Output:
- numpy array of boundary pixels with respect to coordinates given.

### Non-boundary free pixels
To calculate the non-boundary free pixels, the following method is available:
```python
non_b_pixels = dart.non_boundary_free_pixels(boundaries, p)
```
Parameters:
- `boundaries` : is the boundaries binary mask as a binary 2D numpy array, as defined in the output of the method `boundary_pixels`.
- `p` : defines the probability for a pixel to not be sampled as a non-boudary free pixel.

Output:
- The output `non_b_pixels` is a binary 2D matrix, where the True values represent if a given pixel was sampled as a free pixel.

### Algebraic Reconstruction
For the continous reconstruction step, various algorithms have been implemented. Specifically, **SART**, **SIRT**, **ART** and **FBP** are available for experimentation.

The following example demostrates how to use SART:
```python
sart_res_id, sart_res = DART().SART(vol_geom, projector_id, sino_id, iters, use_gpu=True)
```

Parameters:
- `vol_geom`: represents the volume geometry for the output.
- `projector_id`: specifies the projector to use for the measurements.
- `sino_id`: is the sinogram id of the projections.
- `iters`: number of dart iterations to run.
- `use_gpu`: set to True to run Astra on GPU. You also need to use a gpu capable projector.

Output:
- The algorithm will return `sart_res_id` which is the astra-toolbox reference to the reconstructed phantom, and `sart_res`, a numpy array with the actual values of the reconstructed phantom.  

## Examples and Results

**to be added**

## TODO

- add smoothing as last step of dart algorithm
- add sampling intermediate steps of the algorithm
- add getting error for plots

## References

<div id="original_publication">
[1].<br/>
Batenburg, Kees & Sijbers, Jan. (2011). DART: A Practical Reconstruction Algorithm for Discrete Tomography. IEEE transactions on image processing : a publication of the IEEE Signal Processing Society. 20. 2542-53. 10.1109/TIP.2011.2131661, <a href="https://ieeexplore.ieee.org/document/5738333">https://ieeexplore.ieee.org/document/5738333</a>
</div>

<br/>
<div id="astra_1">
[2].<br/>
W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, “Fast and Flexible X-ray Tomography Using the ASTRA Toolbox”, Optics Express, 24(22), 25129-25147, (2016),
 <a href="http://dx.doi.org/10.1364/OE.24.025129">http://dx.doi.org/10.1364/OE.24.025129</a>
</div>

<br/>
<div id="astra_2">
[3].<br/>
W. van Aarle, W. J. Palenstijn, J. De Beenhouwer, T. Altantzis, S. Bals, K. J. Batenburg, and J. Sijbers, “The ASTRA Toolbox: A platform for advanced algorithm development in electron tomography”, Ultramicroscopy, 157, 35–47, (2015), <a href="http://dx.doi.org/10.1016/j.ultramic.2015.05.002">http://dx.doi.org/10.1016/j.ultramic.2015.05.002</a>
</div>

<br/>
<div id="astra_3">
[4].<br/>
W. J. Palenstijn, K. J. Batenburg, and J. Sijbers, “Performance improvements for iterative electron tomography reconstruction using graphics processing units (GPUs)”, Journal of Structural Biology, vol. 176, issue 2, pp. 250-253, 2011, <a href="http://dx.doi.org/10.1016/j.jsb.2011.07.017">http://dx.doi.org/10.1016/j.jsb.2011.07.017</a>
</div>

<br/>
<div id="SART">
[5].<br/>
Yu, Hengyong & Wang, Ge. (2010). SART-Type Image Reconstruction from a Limited Number of Projections with the Sparsity Constraint. International journal of biomedical imaging. 2010. 934847. 10.1155/2010/934847. <a href="https://www.hindawi.com/journals/ijbi/2010/934847/">https://www.hindawi.com/journals/ijbi/2010/934847/</a>
 </div>
