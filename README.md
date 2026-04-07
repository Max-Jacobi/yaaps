# Yet Another Athena Plot Script (YAAPS)

Postprocessing scripts for [GR-Athena++](https://github.com/computationalrelativity/gr-athena).

This repo was forked from commit [74a0b3b](https://github.com/Max-Jacobi/yaaps/commit/74a0b3b256c3abfff21a854b4db0e76397988251) of [yaaps](https://github.com/Max-Jacobi/yaaps).

This repo makes use of the fork [simtroller](https://github.com/Max-Jacobi/simtroller) of the original [simtroller](https://bitbucket.org/bdaszuta/simtroller/src/master/).

Several scripts assume the same folder structure for your [CoRe](https://github.com/computationalrelativity) repos as the one described at in the [grathena-simulations](https://github.com/computationalrelativity/grathena-simulations) repo [guides](https://github.com/computationalrelativity/grathena-simulations/blob/master/guides) (defines e.g. `${DIR_REP}` and `${DIR_CR}`).
Some scripts also assume that your simulation directory is managed by [batchtools](https://github.com/computationalrelativity/batchtools).


## Install

The scripts in `yaaps` work on the following dependencies:
```
Python3.14
numpy
scipy
h5py
matplotlib
tqdm
ffmpeg (with enabled gpl)
```

Clone:
```
cd ${DIR_REP}
git clone git@github.com:Magistrelli/yaaps.git
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

Initialize your simulation directory with [batchtools](https://github.com/computationalrelativity/batchtools).
Example with simulations for [grathena-runs-m1-a](https://github.com/computationalrelativity/grathena-runs-m1-a) projects:
```
batchtools init --batch ${DIR_CR}/batchtools/master/templates/grathena/lrz-gcc/athena.sub --parfile ${DIR_CR}/grathena-runs-m1-a/master/inputs/<path/to/input.inp> --exe ${DIR_CR}/grathena-runs-m1-a/master/exec/<exe> --include ${DIR_REP}/yaaps/analyze.sh --include ${DIR_REP}/yaaps/examples/plot_vars.sh  --include ${DIR_REP}/yaaps/examples/plot_hst.py
ln -s BATCH/include/analyze.sh .
```

The script [analyze.sh](./analyze.sh) combines the available restarts and produces a series of visualisations useful to check the sanity of the run and to make some first analyses,
Run the script with
```
source analyze.sh
```
Run 
```
source analyze.sh -h
```
for usage and more info.

The script [examples/plot_vars.sh](./examples/plot_vars.sh) provides a working example to produce animations and plots from any output file of the simulation.
To run the script, first copy it or link it in the `/path/to/simdir/combine` folder, and then
```
source plot_vars.sh -V
```
to list all the variables saved in the simulation output (original names, some coupled with their more standard names), or e.g.
```
source plot_vars.sh -v <var1> <var2> ...
```
to produce animations of the selected variables.
Run 
```
source plot_vars.sh -h
```
for usage and more info.

The script [examples/plot_hst.py](./examples/plot_hst.py) plots the history of a series of selected scalar variables. To see the available scalar variables run
```
python plot_hst.py -s /path/to/output/folder -l
```

### Other scripts

The script [combine.py](./combine.py) creates a new subfolder `combine` in the simulations directory and combines the outputs from all the restart folders, which are assumed to be called `output-####` (same structure of the simulation folder automatically created by [batchtools](https://github.com/computationalrelativity/batchtools).
Copy the script or link it in the `/path/to/simdir/combine` folder.
Then, run
```
python combine.py .
```
