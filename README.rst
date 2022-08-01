.. image:: https://raw.githubusercontent.com/oskarfernlund/orthogonal-svgp/master/figures/logo.png
    :width: 500
    
Orthogonal sparse variational Gaussian processes (SVGP's) in TensorFlow.

Orthogonal methods [1][2] extend standard SVGP approaches [3] by splitting the 
inducing points learned by the model into two orthogonal sets with structured 
covariance. The resulting approximate posterior can be decomposed into several 
independent (orthogonal) Gaussian processes and both training and prediction 
involves reduced complexity matrix inversions as compared with standard SVGP. 
This project implements an orthogonal SVGP framework using GPflow/TensorFlow 
and explores whether orthogonal methods are indeed able to provide high quality 
posterior approximations with fewer inducing points than standard SVGP methods 
and/or at reduced computational cost.

[1] https://arxiv.org/pdf/1809.08820.pdf
[2] https://arxiv.org/pdf/1910.10596.pdf
[3] https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf


Dependencies
------------

This project relies heavily on GPflow_, as well as NumPy, Matplotlib, Jupyter, 
TensorFlow and TensorFlow Probability. Note that TensorFlow Probability 
releases are tightly coupled to TensorFlow; please ensure that the TensorFlow 
and TensorFlow Probability versions you have installed are compatible. You may 
wish to manually install an older version of TensorFlow Probability to address 
compatibility issues. Properly installing TensorFlow depends on your operating 
system and hardware, but a sample ``environment.yml`` file containing complete 
project dependencies may be found in the ``env/`` directory which can be used 
to set up the environment with Anaconda for M1 Mac users.

.. code-block:: console

    $ cd env/
    $ conda env create -f environment.yml

.. _GPflow: https://www.gpflow.org/


Contributors & Acknowledgements
-------------------------------

All source code was written by Oskar Fernlund at Imperial College London in 
support of his 2022 Master's thesis. Implementation of the orthogonal framework 
depends heavily on GPflow and roughly follows its model API. For a complete 
list of GPflow contributors, see `GPflow's GitHub page`_. Many thanks to Mark 
van der Wilk and Anish Dhir for all their insightful feedback and support. 

.. _`GPflow's GitHub page`: https://github.com/GPflow/GPflow/


License
-------

Use and distribution of the source code is covered by the Apache-2.0 license.