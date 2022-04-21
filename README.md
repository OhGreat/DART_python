# DART - python
DART (Discrete Algebric Reconstruction Technique) is an iterative reconstruction algorithm for discrete tomography.

This repository contains an implementation of the DART algorithm, together with a framework to generate phantoms and data, to test the algorithm itself.

## Prerequisites

`Python 3` and the following packages are required to use this library:

- `numpy`
- `Pillow`
- `foam_ct_phantom` : documentation is available <a href="https://github.com/dmpelt/foam_ct_phantom">here</a>.
- `astra-toolbox` : documentation is available <a href="https://www.astra-toolbox.com/">here</a>.

## Usage

### Generating phantoms

<b>Foam phantoms</b><br/>
To generate foam like phantoms the library `foam_ct_phantom` was used. The file `generate_foam.py` can be used to create similar phantoms to the ones used in our experiments by running the command:

```
python generate_foam.py
```

*(a bash script to create phantoms and set arguments will be included in the future)*

### Generating projection angles and detector values

### Running DART

## Examples and Results