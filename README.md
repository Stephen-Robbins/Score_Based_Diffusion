# Score Based Diffusion

## Overview
This repository contains our implementation of score-based diffusion models, developed as part of the course 274E. The project focuses on exploring score-based diffusion processes on 2D data sets, specifically MNIST and CIFAR-10. Beyond the standard score-based diffusion generative models, we look at various applications and extensions, including conditional diffusion, infilling, and diffusion bridges.

## Team Members
- [Tomas Ortega](https://github.com/TomasOrtega)
- [Kai Nelson](https://github.com/KaiTyrusNelson)

## Project Structure
- `data.py`: Two dimensional data sets.
- `diffusion.py`: Implementation of the forward and backward diffusion processes.
- `guided_diffusion.py`: Implementation of guided diffusion techniques.
- `model.py`: Neural network models to approximate the score.
- `training.py`: Training scripts for the diffusion models.
- `config.yaml`: Configuration file for model and training parameters.
- `example_notebooks/`: Jupyter notebooks demonstrating the usage and results of our models.

## Acknowledgments
This project was a collaborative effort as part of the course 274E. We would like to thank our course instructors and colleagues for their support and guidance throughout the development of this project.

To play with the code, see the Jupyter notebook `examples.ipynb`.

It is highly recommended to use a GPU for training. If you do not have a GPU, you can use [this MNIST example on Google Colab](https://colab.research.google.com/drive/1e2G_uPZiRbOl2s9oCuVhSptL_Qgb4nLy?usp=sharing).
