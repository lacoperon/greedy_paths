# Greedy Lattice Test Code

The goal of this collection of code snippets is to reproduce the results in Kleinberg's 'Navigation in a small world' (2000), generalized to different of shortcuts being generated, as well as to higher dimension lattices.

The Nature Brief Communications pdf is in the folder; the simulation code
is in `greedy_lattice_test.py`, and the code that collects the data and graphs
it lives in `dataanalysis.Rmd` (this is not quite done yet).

## Workflow

To run a simulation, edit the simulation parameters (found at the end of
`greedy_lattice_test.py`) and type `python greedy_lattice_test.py` into
your command line client of choice.

To analyze the data from each run, open up `dataanalysis.Rmd` in RStudio and
run all of the code blocks (**TODO**: Currently the .Rmd is not generalized
to work for all simulation parameters).

(**Also TODO:** Add a bash script which knits the .Rmd document without opening
up RStudio -- this would work much better in practice and would speed up running
simulations)

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

TODO: (21st December, 2017)

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

Thoughts from Elliot (Dec 23rd):

Do we want to collect the size of the backwards step?
I'm using Manhattan distance between successive nodes and the trg
to quantify what a backwards step is, but that also seems to imply
that there are backwards steps that are more backwards than others.

again, do we want to collect the size of these steps? What does this
actually look like in practice
