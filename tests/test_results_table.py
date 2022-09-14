import os
import tempfile
from unittest import TestCase

from hdmf.common import HDF5IO, get_manager, get_hdf5io, EnumData
from hdmf_ml import ResultsTable

import numpy as np


class ResultsTableTest(TestCase):

    def setUp(self):
        _, self.path = tempfile.mkstemp()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def get_hdf5io(self):
        #return get_hdf5io(self.path, 'w')
        return HDF5IO(self.path, 'w', manager=get_manager())


    # add_tvt_split
    # add_cv_split
    # add_true_label
    # add_predicted_probability
    # add_predicted_class
    # add_predicted_value
    # add_cluster_label
    # add_embedding
    # add_topk_classes
    # add_topk_probabilities

    def test_ResultsTable_tvt_split(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_tvt_split([0, 1, 2, 0, 1])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_cv_split(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_cv_split([0, 1, 2, 3, 4])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_true_label(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_true_label([0, 0, 0, 1, 1])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_true_label_str(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_true_label(['a', 'a', 'a', 'b', 'b'])
        self.assertIsInstance(rt['true_label'], EnumData)
        np.testing.assert_array_equal(rt.true_label.elements.data, np.array(['a', 'b']))
        with self.get_hdf5io() as io:
            io.write(rt)


    def test_ResultsTable_predicted_probability(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_predicted_probability([[0.1, 0.9],
                                      [0.2, 0.8],
                                      [0.3, 0.7],
                                      [0.4, 0.6],
                                      [0.5, 0.5]])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_predicted_class(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_predicted_class([0, 0, 1, 1, 1])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_predicted_value(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_cv_split([0.0, 0.1, 0.2, 0.3, 0.4])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_cluster_label(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_cv_split([0, 1, 2, 1, 0])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_embedding(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_embedding([[1.1, 2.9],
                          [1.2, 2.8],
                          [1.3, 2.7],
                          [1.4, 2.6],
                          [1.5, 2.5]])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_topk_classes(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_topk_classes([[1, 2],
                             [3, 4],
                             [5, 6],
                             [7, 8],
                             [9, 0]])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_ResultsTable_topk_probabilities(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_topk_probabilities([[0.9, 0.1],
                                   [0.8, 0.2],
                                   [0.7, 0.3],
                                   [0.6, 0.4],
                                   [0.5, 0.5]])
        with self.get_hdf5io() as io:
            io.write(rt)
