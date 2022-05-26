# Discrete Algebric Reconstruction Technique (DART) - *currently in development*
DART is an iterative reconstruction algorithm for discrete tomography. The original publication in <a href="#original_publication">[1]</a> was used as reference to create this library.
What this repository consists of, is an implementation of the DART algorithm together with a framework to generate phantoms and measurements, to test the algorithm itself.

## DART

### ART reconstruction step
The DART algorithm, implemented as in the original publication <a href="#original_publication"> [1]</a>, alternates between continuous and discrete reconstruction steps. For the continuous step, a variant of the SART algorithm is used. For the original publication of SART you can refer to <a href="#SART">[5]</a>.

## Prerequisites

`Python 3` and the following packages are required to use this library:

- `numpy`
- `Pillow`
- `foam_ct_phantom` : required to create phantoms, documentation is available <a href="https://github.com/dmpelt/foam_ct_phantom">here</a>.
- `astra-toolbox` : required to create phantoms and projections, documentation is available <a href="https://www.astra-toolbox.com/">here</a>. Publications relevant to this library can be found in <a href="#astra_1">[2]</a>, <a href="#astra_2">[3]</a> and <a href="#astra_3">[4]</a>.

## Usage
To run DART, data *(in the form of phantoms)* and measurements *(in the form of projections and detector values)* need to be artificially constructed. 
Therefore, three main wrappers have been created around the code to solve the following tasks:
- phantom creation
- projection and measurements acquisition
- running DART reconstruction algorithm

Usage of the framework for each of this tasks is described in detail in the following sections.

### Generating phantoms

<b>Foam phantoms</b><br/>
To generate foam like phantoms the library `foam_ct_phantom` was used. The main function to create foam like phantoms is `create_foam` in the `phantoms.foam` package. To import it and use it run the following commands:
```python
from phantoms.foam import create_foam
create_foam(filename, n_sheres, seed)
```
where the function parameters are the following:
- `filename`: name of the file to save. To be passed as a string.
- `n_spheres`: iterations of the algorithm. The more iterations, the more 'holes' the foam phantom will have. Default value is 1000. 
- `seed`: integer value that sets the random generated values, used to reproduce results.

The function does not return any value, you can proceed directly to loading the generated phantom with the astra toolkit.

### Generating projections

#### From 2D phantoms
To generate measurements in the form of 1D projections, the function `project_from_2D` has been created. You can import it and use it with the following commands:

```python
from measurements.projections import project_from_2D
proj_id, projections = project_from_2D(phantom_data, n_projections, detector_spacing, apply_noise=False, save_dir=None)
```
where:
- `phantom_data`: is the phantom as a 2D numpy array.
- `n_projections`: is an integer value representing the number of projections as the number of angles to make measurements from.
- `detector_spacing`: defines the size of the pixel.
- `apply_noise`: boolean value that adds Poisson distributed noise to the image when set to True. False by default.
- `save_dir`: string representing the directory to save png images that represent the measurements. Images won't be saved if this parameter is not set.

The function will return `proj_id` and `projections`. The first is a reference to the astra toolkit object while the second one is a vector with the measurements itself.

#### From 3D phantoms

### Running DART

**to be completed**
All the steps required to run the DART algorithm have been broken down and can be used separately. A detailed desctiption for the usage of all the functions available in the DART library will therefore follow in this section.
 
#### Instanciating DART
In order to use the package and all of the methods, DART has to be imported and instanciated as follow:
```python
from reconstruction_algs.DART import DART
dart_instance = DART()
```
### Segmentation
The method `segment` can be used to segment an image, given the range of gray values:
```python
segmented_img = dart.segment(img, gray_levels)
```
where:
- `img`: is the grayscale input phantom to segment as a 2D numpy matrix.
- `gray_levels` : array of gray levels to compute the thresholds for the segmentation from.

### Pixel neighborhood
To calculate the indexes of neighbours of a specific pixel, you can use the method `pixel_neighborhood` as below:
```python
neighbours = dart.pixel_neighborhood(img, x, y)
```
where:
- `img` : is as usual the phantom in the form of 2D numpy matrix.
- `x,y`: are the coordinates of the pixel in 

The method returns a 2D array containing arrays of the x,y coordinates of the neighbours.

### Boundary pixels
To calculate the boundary pixels of the phantom image, the method `boundary_pixels` takes as input the phantom *as a numpy array*, and calculates the boundary pixels with the help of the `pixel_neighborhood` method described above. You can use it as follows:
```python
b_pixels = dart.boundary_pixels(img):
```
where, `b_pixels` is the output given as a 2D binary mask of the image, where True values represent that the given pixel is a boundary pixel.

### Non-boundary free pixels
To calculate the non-boundary free pixels, the following method is available:
```python
non_b_pixels = dart.non_boundary_free_pixels(boundaries, p)
```
where:
- `boundaries` : is the boundaries binary mask as a binary 2D numpy array, as defined in the output of the method `boundary_pixels`.
- `p` : defines the probability for a pixel to not be sampled as a non-boudary free pixel.

The output `non-b_pixels` is yet again a binary 2D matrix, where the Tru values represent if a given pixel was sampled as a free pixel.

## Examples and Results

**to be added**

## TODO

- add smoothing as last step of dart algorithm
- 

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
