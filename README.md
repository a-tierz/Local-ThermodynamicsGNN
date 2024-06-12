
<div align="center"> 

# Graph neural networks informed locally by thermodynamics

[![Project page](https://img.shields.io/badge/-Project%20page-blue)](https://amb.unizar.es/people/alicia-tierz/)

</div>

## Abstract
Thermodynamics-informed neural networks employ inductive biases for the enforcement of the first and second principles of thermodynamics. To construct these biases, a metriplectic evolution of the system is assumed. This provides excellent results, when compared to uninformed, black box networks. While the degree of accuracy can be increased in one or two orders of magnitude, in the case of graph networks, this requires assembling global Poisson and dissipation matrices, which breaks the local structure of such networks. In order to avoid this drawback, a local version of the metriplectic biases has been developed in this work, which avoids the aforementioned matrix assembly, thus preserving the node-by-node structure of the graph networks. We apply this framework for examples in the fields of solid and fluid mechanics.  Our approach demonstrates significant computational efficiency and strong generalization capabilities, accurately making inferences on examples significantly different from those encountered during training.

For more information, please refer to the following:

- Tierz, Alicia and Alfaro, Iciar and González, David and Chinesta, Francisco and Cueto, Elías. "[Graph neural networks informed locally by thermodynamics](https://ieeexplore.ieee.org/document)." IEEE Transactions on Artificial Intelligence (2024).


<div align="center">
<img src="/outputs/epoch_150.gif" width="450"><img src="/outputs/result_pdc.gif" width="350">
</div>


## Setting it up

First, clone the project.

```bash
# clone project
git clone https://github.com/a-tierz/Local-ThermodynamicsGNN
cd ThermodynamicsGNN
```

Then, install the needed dependencies. The code is implemented in [Pytorch](https://pytorch.org). _Note that this has been tested using Python 3.9_.

```bash
# install dependencies
pip install numpy scipy matplotlib torch torch-geometric torch-scatter
 ```

## How to run the code  

### Test pretrained nets