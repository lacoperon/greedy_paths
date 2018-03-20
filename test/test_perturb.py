import sys, os
import pytest
import networkx as nx
import random
from functools import reduce

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import perturbation_approaches as pr


class TestGraphGeneration(object):
    def test_lattice1(self):
        G = pr.generateGraph("lattice", N=1000, dim=1)
        assert len(G.nodes()) == 1000
        assert len(G.edges()) == 999
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_lattice2(self):
        G = pr.generateGraph("lattice", N=6400, dim=1)
        assert len(G.nodes()) == 6400
        assert len(G.edges()) == 6399
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_lattice3(self):
        G = pr.generateGraph("lattice", N=1000, dim=2)
        assert len(G.nodes()) == 961
        assert len(G.edges()) == 1860
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 4

    def test_lattice4(self):
        G = pr.generateGraph("lattice", N=6400, dim=2)
        assert len(G.nodes()) == 6400
        assert len(G.edges()) == 12640
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 4

    def test_lattice5(self):
        G = pr.generateGraph("lattice", N=2000, dim=3)
        assert len(G.nodes()) == 1728
        assert len(G.edges()) == 4752
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 6

    def test_lattice6(self):
        G = pr.generateGraph("lattice", N=6400, dim=3)
        assert len(G.nodes()) == 5832
        assert len(G.edges()) == 16524
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 6

    def test_geometric1(self):
        G = pr.generateGraph("geometric", N=1000, dim=1, k=2)
        assert len(G.nodes()) == 1000
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_geometric2(self):
        G = pr.generateGraph("geometric", N=6400, dim=1, k=2)
        assert len(G.nodes()) == 6400
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_geometric3(self):
        G = pr.generateGraph("geometric", N=1000, dim=2, k=2)
        assert len(G.nodes()) == 1000
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_geometric4(self):
        G = pr.generateGraph("geometric", N=6400, dim=2, k=2)
        assert len(G.nodes()) == 6400
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_geometric5(self):
        G = pr.generateGraph("geometric", N=1000, dim=3, k=2)
        assert len(G.nodes()) == 1000
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2

    def test_geometric6(self):
        G = pr.generateGraph("geometric", N=6400, dim=3, k=2)
        assert len(G.nodes()) == 6400
        degrees = nx.average_neighbor_degree(G).values()
        ave_degree = reduce(lambda x,y : x + y, degrees) / len(degrees)
        assert int(round(ave_degree)) == 2
