# Greedy Lattice Test Code

The goal of this collection of code snippets is to reproduce the results in
Kleinberg's 'Navigation in a Small World' paper, generalized to more varieties
of network perturbations, as well as to higher-dimensional lattices.

The simulation code is contained in `greedy_lattice_test.py`, and `dataanalysis.Rmd`
parses the csv files produced from each run, and graphs various measures of
interest (this is not quite done yet).

## Installation / Setup

First of all, the project runs Python 2.7, so ensure you have the right version
installed.

To install the dependencies for the project, run `pip install -r requirements.txt`
in the Command Line.

## Workflow

To run a simulation, edit the simulation parameters (found at the end of
`greedy_lattice_test.py`) and type `python greedy_lattice_test.py` into
your command line client of choice. The data outputted from each run is
put into the `data_output` folder in csv format.

To open the analysis of data from each run, open up `DataAnalysisJupyter.ipynb`,
by typing in `jupyter notebook DataAnalysisJupyter.ipynb`.

Note that the Jupyter Notebook is written in R, which is not available as a
default language for Jupyter. In an ideal world, I would have setup the project
to automatically install this for new users, or I would have a Docker environment
that I could provide in the GitHub with Jupyter with R integration already installed.

I didn't bother (because I'm, thus far, the only one actually running this code).

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

Connection w how good is an embedding

try to count of backwards steps in a path, for each path
also:   

* steps for k = 1
* steps for k = 2
* ...
* backwards steps for k > 1 (separated by case)
* shortest path length for the src, trg
* shortcuts taken (k > 1)
* shortcuts taken (k = 1)
* shortcuts taken in shortest path

instead of sampling pairs, just sample src maybe and use all such
targets maybe

curve w steps, alphas, w avg + std

**Thoughts from Elliot (Dec 23rd):**

Do we want to collect the size of the backwards step?
I'm using Manhattan distance between successive nodes and the trg
to quantify what a backwards step is, but that also seems to imply
that there are backwards steps that are more backwards than others.

again, do we want to collect the size of these steps? What does this
actually look like in practice

*NOTES ON THIS (Jan 18th)*: We might want the maximum sized backwards step,
and all of the sizes of backwards steps (maybe just max and sum)

**Thoughts from Elliot (Jan 4):**

~How many cores does the computer we're gonna run this on have?
(ie is it worth parallelizing everything, down to the src trg stuff for each
 graph? Yes if a huge # of cores, no if the grain of alpha is enough to be a
 rate limiting step).

 --> AKA yes, if running on wesleyan HPC cluster

*NOTES ON THIS (Jan 18th)*: Do runs for same n with different dimension,
see if the distribution of path lengths is different/more broad/narrow.
TEST THIS!!!

 ALSO, for the graphs, don't we want the bars to be boostrapped std dev of the
 actual run average length, as opposed to the std dev of the paths for one particular run?
 The former seems like it would converge for many iterations, whereas the latter doesn't.

 **Notes on Meeting (Jan 18):**
 Should scale pair_frac with N value.
 Write down which version of everything we're using.
 (IE which version of networkx is running) >>> 1.11
 https://stackoverflow.com/questions/5226311/installing-specific-package-versions-with-pip
 ^^^ this is how to  install specific version of pip python modules

 ALSO, should also use a virtualenv to keep packages local to projects
 https://www.quora.com/What-is-difference-between-pip-and-pip3

 Also, should probably have stuff being logged as runs go on -->
 allows for checking of runtimes as the cluster runs

 Try Jupyter for visualizing output. Add unit tests to make code much more
 tolerable to changes.
