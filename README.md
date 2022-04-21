# Discrete Algebric Reconstruction Technique (DART) - *currently in development*
DART is an iterative reconstruction algorithm for discrete tomography. The original publication<a href="#original_publication">[1]</a> was used as reference to create this library.

What this repository consists of is an implementation of the DART algorithm, together with a framework to generate phantoms and measurements, to test the algorithm itself.

## Prerequisites

`Python 3` and the following packages are required to use this library:

- `numpy`
- `Pillow`
- `foam_ct_phantom` : documentation is available <a href="https://github.com/dmpelt/foam_ct_phantom">here</a>.
- `astra-toolbox` : documentation is available <a href="https://www.astra-toolbox.com/">here</a>. Publications relevant to this library are <a href="astra_1">[2]</a>, <a href="astra_2">[3]</a> and <a href="astra_3">[4]</a>.

## Usage
To run DART, data *(in the form of phantoms)* and measurements *(in the form of projections and detector values)* need to be artificially constructed. 
Three main wrappers have been created around the code to solve the following tasks:
- phantom creation
- projection and measurements acquisition
- running DART reconstruction algorithm

Usage of the framework for each of this tasks is described in detail in the following sections.

### Generating phantoms

<b>Foam phantoms</b><br/>
To generate foam like phantoms the library `foam_ct_phantom` was used. The file `generate_foam.py` can be used to create similar phantoms to the ones used in our experiments by running the command:

```
python generate_foam.py
```

*(a bash script to create phantoms and set arguments will be included in the future)*

### Generating projection angles and detector values

*to be added soon*

### Running DART

*to be added soon*

## Examples and Results

*to be added soon*

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
