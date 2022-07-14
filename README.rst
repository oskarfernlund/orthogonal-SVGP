.. image:: https://raw.githubusercontent.com/oskarfernlund/orthogonal-svgp/master/figures/logo.png
    :width: 500
    
Orthogonal sparse variational Gaussian processes (SVGP's) in TensorFlow.

Orthogonal SVGP methods extend the standard SVGP model by splitting the 
inducing points into two sets with structured covariance. The result is an 
approximate posterior which can be decomposed into several independent Gaussian 
processes and which involves reduced complexity matrix inversions as compared 
with standard SVGP. This project implements orthogonal SVGP methods using 
TensorFlow and explores whether they are able to provide high quality posterior 
approximations with fewer inducing points than standard SVGP methods. 

- https://arxiv.org/pdf/1809.08820.pdf
- https://arxiv.org/pdf/1910.10596.pdf


Dependencies
------------

This project is built on a forked version of GPflow_ and relies on NumPy, 
SciPy, Matplotlib, Jupyter, TensorFlow, TensorFlow Probability, as well as 
tabulate, lark, deprecated and multipledispatch. Note that TensorFlow 
Probability releases are tightly coupled to TensorFlow; please ensure that the 
TensorFlow and TensorFlow Probability versions you have installed are 
compatible. You may wish to manually install an older version of TensorFlow 
Probability to address compatibility issues. Properly installing TensorFlow 
depends on your operating system and hardware, but a sample ``environment.yml`` 
file containing complete project dependencies may be found in the ``env/`` 
directory which can be used to set up the environment with Anaconda for M1 Mac 
users.

.. code-block:: console

    $ cd env/
    $ conda env create -f environment.yml

.. _GPflow: https://www.gpflow.org/


Contributors & Acknowledgements
-------------------------------

For a complete list of GPflow contributors, see `GPflow's GitHub page`_. 
Extensions pertaining to orthogonal SVGP methods were written by Oskar Fernlund 
at Imperial College London in support of his 2022 Master's thesis. Many thanks 
to Mark van der Wilk and Anish Dhir for all their insightful feedback and 
support.

.. _`GPflow's GitHub page`: https://github.com/GPflow/GPflow/


License
-------

Use and distribution of the source code is covered by the Apache-2.0 license.