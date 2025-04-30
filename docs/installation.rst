WEIS Installation
=================

To install WEIS, please follow the up-to-date instructions contained in the README.md at the root level of this repo, or on the `WEIS GitHub page <https://github.com/WISDEM/WEIS/>`_.

Installing SNOPT for use within WEIS
------------------------------------
SNOPT is available for purchase `here 
<http://www.sbsi-sol-optimize.com/asp/sol_snopt.htm>`_. Upon purchase, you should receive a zip file. Within the zip file, there is a folder called ``src``. To use SNOPT within WEIS, paste all files from ``src`` except snopth.f into ``WEIS/pyoptsparse/pyoptsparse/pySNOPT/source``.
If you do this step before you install WEIS, SNOPT will be automatically compiled within pyOptSparse and should be usable.
Otherwise, you can simply re-install WEIS following the same installation instructions after removing all ``build`` directories from the WEIS, WISDEM, and pyOptSparse directories.


Installing julia and OWENS for vawt or crossflow within WEIS
------------------------------------------------------------
Note that the installation has only been tested on MacOS.

    1. Install julia v1.11 ``curl -fsSL https://install.julialang.org | sh``. If julia is downloaded and installed manually, export the PATH to julia, for example `export PATH="$PATH:/Applications/Julia-1.10.app/Contents/Resources/julia/bin"`. Make sure julia can be called from terminal by typing ``julia``. 
    2. Install WEIS-CTOpt and the correct version of wisdem. 
    3. Activate your weis conda environment. Run the `installation.py` script in OWENS example directory ``06_owens_opt``.
    4. If you get errors complaining about hdf5 when you run the example, check if you have hdf5 in your conda env, if you do, uninstall it ``conda uninstall hdf5`` because it conflicts with julia hdf5 library and then ``pip install h5py``.