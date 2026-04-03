# Yet Another Athena Plot Script (YAAPS)

Postprocessing scripts for [GR-Athena++](https://github.com/computationalrelativity/gr-athena).

This repo was forked from commit [74a0b3b](https://github.com/Max-Jacobi/yaaps/commit/74a0b3b256c3abfff21a854b4db0e76397988251) of [yaaps](https://github.com/Max-Jacobi/yaaps).

This repo makes use of the fork [simtroller](https://github.com/Max-Jacobi/simtroller) of the original [simtroller](https://bitbucket.org/bdaszuta/simtroller/src/master/).


## Install

The scripts in `yaaps` work on the following dependencies:
```
Python3.14
numpy
scipy
h5py
matplotlib
tqdm
simtroller
```

`yaaps` can only be installed with `pip install .` if the machine has free access to internet (e.g. no cluster firewall). If this is not the case, `yaaps` can be set with
```
cd yaaps
pip install --no-build-isolation .
```
and by adding
```
export PYTHONPATH=$PYTHONPATH:${DIR_REP}/yaaps
```
to your `~/.bashrc`.


## Run

The script [combine.py](combine.py) creates a new subfolder `combine` in the simulations directory and combines the outputs from all the restart folders, which are assumed to be called `output-####` (same structure of the simulation folder automatically created by [batchtools](https://github.com/computationalrelativity/batchtools).

To use the script [examples/yaaps.sh](./examples/yaaps.sh), which provides a working example to produce animations and plots from any output file of the simulation, first copy it or link it in the `/path/to/simdir/combine` folder.
Then, run
```
source yaaps.sh -V
```
to list all the variables saved in the simulation output (original names, some coupled with their more standard names), or e.g.
```
source yaaps.sh -v <var1> <var2> ...
```
to produce animations of the selected variables.
Run 
```
source yaaps.sh -h
```
for usage and more info.
