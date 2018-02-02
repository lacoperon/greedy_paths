# Greedy Lattice Test Code

[![Build Status](https://travis-ci.com/lacoperon/greedy_paths.svg?token=Wy9uNtocawbdbMCz8QoS&branch=master)](https://travis-ci.com/lacoperon/greedy_paths)

The goal of this collection of code snippets is to reproduce the results in
Kleinberg's 'Navigation in a Small World' paper, generalized to more varieties
of network perturbations, as well as to higher-dimensional lattices.

The simulation code is contained in `greedy_lattice_test.py`, and `dataanalysis.Rmd`
parses the csv files produced from each run, and graphs various measures of
interest (this is not quite done yet).

## Installation / Setup

You can use the conda package to set up the relevant environment,
with all dependencies already installed, for this project.

First of all, if you don't have it already, you can install conda
(ie through `sudo apt-get conda`, `brew install conda` or looking up a tutorial online).

Then, run `conda env create` to create the `greedy` environment I've defined in
`environment.yml` for this project.  

Then, activate the environment using `source activate greedy`
on macOS and Linux (`activate greedy` in Windows). The conda environment includes
all packages necessary for the code proper, the unit testing, and the Jupyter
Notebook.

## Workflow

To run a simulation, edit the simulation parameters (found at the end of
`greedy_lattice_test.py`) and type `python greedy_lattice_test.py` into
your command line client of choice. The data outputted from each run is
put into the `data_output` folder in csv format.

To open the analysis of data from each run, open up the notebook with the data
analysis code by typing in `jupyter notebook DataAnalysisJupyter.ipynb`.

---

## Various Unstructured Notes
#### (not guaranteed to be coherent...)

See how greedy paths are doing (compared to actual shortest paths)
compare paths generated to actual shortest paths (nx implements this),
and then see that compared for various small n, alpha, p

see how the not-so-greedy path are doing

Start based on line graph; try to recover the results,
then try to generalize to more dimensions

i.e. optimal alpha should be 1 for linear,
                             2 for 2D lattice, etc

For each node, with probability p, add a shortcut based on the rules
we have n-2 nodes (linear case) we're not connected to, how do we choose?
get a bunch of distances, apply the ** - alpha, pick a shortcut proportional
to value in the rs_alpha array, then run this simulation for many alphas
and plot (Kleinberg has p=1, so try with this first)

**Notes from Meeting on 21st December, 2017**

Count to see the number of weird cases -- where the num=k case leads to a
shorter path when compared to the num=k+1 case, and insodoing see how the
graph will potentially change.

Two types of effects -- true effect (IE k=2 is slower, deterministically,
as compared to k=1) -- vs. "bug" effect, which is possiblity random choices
make 'large' (or any) differences in the shortest path length computed
(bug effect is only for d>1)


**Thoughts from Elliot (Dec 23rd):**

Do we want to collect the size of the backwards step?
I'm using Manhattan distance between successive nodes and the trg
to quantify what a backwards step is, but that also seems to imply
that there are backwards steps that are more backwards than others.

again, do we want to collect the size of these steps? What does this
actually look like in practice

*NOTES ON THIS (Jan 18th)*: We might want the maximum sized backwards step,
and all of the sizes of backwards steps (maybe just max and sum)


*NOTES ON THIS (Jan 18th)*: Do runs for same n with different dimension,
see if the distribution of path lengths is different/more broad/narrow.
TEST THIS!!!

 ALSO, for the graphs, don't we want the bars to be boostrapped std dev of the
 actual run average length, as opposed to the std dev of the paths for one particular run?
 The former seems like it would converge for many iterations, whereas the latter doesn't.

 **Notes on Meeting (Jan 18):**
 Should scale pair_frac with N value.

 Also, should probably have stuff being logged as runs go on -->
 allows for checking of runtimes as the cluster runs


**Notes from Feb 2nd**:
Should add flag for doing shortest paths -- probably don't want to for large N
Should also add histogram collection code -- easy way of reducing data produced
(also collect averages -- will otherwise be biased by data degradation by histogram)

Also update writing code so that it writes every run (graph done)
Also time code running for various functions -- helpful to see bottlenecks
