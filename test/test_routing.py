import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import greedy_lattice_test as gr
import pytest
import networkx as nx
import random


class TestHelperFunctions(object):

    def test_manhattan(self):
        assert gr.lattice_dist(1,2) is 1
        assert gr.lattice_dist(13,13) is 0
        assert gr.lattice_dist((1,2,3),(4,5,6)) is 9
        assert gr.lattice_dist((1,2,3),(1,1,1)) is 3
        assert gr.lattice_dist((1,2,3,4,5),(1,2,3,4,5)) is 0

    def test_denovo_adj(self):
        assert gr.are_denovo_adj(1,2) is True
        assert gr.are_denovo_adj(1,1) is False
        assert gr.are_denovo_adj(13,15) is False
        assert gr.are_denovo_adj((1,1),(1,2))

    def test_shortcuts_taken(self):
        path1 = [1,2,3,4,7,8,9]
        assert gr.shortcuts_taken(path1) is 1
        path2 = [1,2,4,5,7,8,10,11]
        assert gr.shortcuts_taken(path2) is 3
        path3 = [(1,1),(1,2),(2,2),(17,3),(17,4)]
        assert gr.shortcuts_taken(path3) is 1
        path4 = []
        assert gr.shortcuts_taken(path4) is 0

    def test_num_backsteps(self):
        G1 = nx.grid_graph([10], periodic=False)
        path1 = [1,2,3,4,7,8,9]
        assert gr.calc_num_backwards_steps(G1, path1, path1[-1]) is 0
        path2 = [1,3,2,4,3,5,4,9]
        assert gr.calc_num_backwards_steps(G1, path2, path2[-1]) is 3

        G2 = nx.grid_graph([20,10], periodic=False)
        path3 = [(1,1),(1,2),(2,2),(2,1),(17,3),(17,4)]
        assert gr.calc_num_backwards_steps(G2, path3, path3[-1]) is 1
        # path4 = []
        # assert gr.shortcuts_taken(path4) is 0


class TestGraphGeneration(object):

    def test_grid_graph_onedim(self):
        for i in range(100):
            grid_input = [10]
            G = nx.grid_graph(grid_input, periodic=False)
            G = G.to_directed()
            assert G.is_directed()
            assert len(G.nodes()) == 10
            assert len(G.edges()) == 18

            for node in G.nodes():
                node_edges = G.edges(nbunch=[node])
                assert 1 <= len(node_edges) and 2 >= len(node_edges)

            G = gr.add_shortcuts(G, p=1, alpha=3*random.random(), verbose=False)
            assert len(G.nodes()) == 10
            assert len(G.edges()) == 28

            for node in G.nodes():
                node_edges = G.edges(nbunch=[node])
                assert 2 <= len(node_edges) and 3 >= len(node_edges)

    def test_grid_graph_twodim(self):
        for i in range(10):
            grid_input = [10,10]
            G = nx.grid_graph(grid_input, periodic=False)
            G = G.to_directed()
            assert G.is_directed()
            assert len(G.nodes()) == 100
            assert len(G.edges()) == 360

            for node in G.nodes():
                node_edges = G.edges(nbunch=[node])
                assert 2 <= len(node_edges) and 4 >= len(node_edges)

            G = gr.add_shortcuts(G, p=1, alpha=3*random.random(), verbose=False)
            assert len(G.nodes()) == 100
            assert len(G.edges()) == 460

            for node in G.nodes():
                node_edges = G.edges(nbunch=[node])
                assert 3 <= len(node_edges) and 5 >= len(node_edges)

    def test_grid_graph_threedim(self):
        grid_input = [10,10,10]
        G = nx.grid_graph(grid_input, periodic=False)
        G = G.to_directed()
        assert G.is_directed()
        assert len(G.nodes()) == 1000
        assert len(G.edges()) == 5400

        for node in G.nodes():
            node_edges = G.edges(nbunch=[node])
            assert 2 <= len(node_edges) and 6 >= len(node_edges)

        G = gr.add_shortcuts(G, p=1, alpha=3*random.random(), verbose=False)
        assert len(G.nodes()) == 1000
        assert len(G.edges()) == 6400

        for node in G.nodes():
            node_edges = G.edges(nbunch=[node])
            assert 3 <= len(node_edges) and 7 >= len(node_edges)
