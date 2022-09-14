import os
import tempfile
from unittest import TestCase

from hdmf.common import HDF5IO, get_manager
from hdmf_ml import ResultsTable


class ResultsTableTest(TestCase):

    def setUp(self):
        _, self.path = tempfile.mkstemp()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def get_hdf5io(self):
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

    def test_ResultsTable(self):
        rt = ResultsTable('foo', 'a test results table')
        rt.add_tvt_split([0, 1, 2, 0, 1])
        rt.add_true_label([0, 0, 0, 1, 1])
        with self.get_hdf5io() as io:
            io.write(rt)

