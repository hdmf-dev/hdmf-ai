import os
import tempfile
from unittest import TestCase

from hdmf.common import HDF5IO, get_manager, EnumData
from hdmf_ml import ResultsTable

import numpy as np


def get_temp_filepath():
    # On Windows, h5py cannot truncate an open file in write mode.
    # The temp file will be closed before h5py truncates it and will be removed during the tearDown step.
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.close()
    return temp_file.name


class ResultsTableTest(TestCase):
    def setUp(self):
        self.path = get_temp_filepath()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def get_hdf5io(self):
        return HDF5IO(self.path, "w", manager=get_manager())

    def test_add_col_diff_len(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_tvt_split([0, 1, 2, 0, 1])
        msg = (
            "New column true_label of length 4 is not the same length as "
            "existings columns of length 5"
        )
        with self.assertRaisesRegex(ValueError, msg):
            rt.add_true_label([0, 0, 0, 1])

    def test_add_col_dupe_name(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_tvt_split(np.uint([0, 1, 2, 0, 1]))
        msg = "Column 'tvt_split' already exists in ResultsTable 'foo'"
        with self.assertRaisesRegex(ValueError, msg):
            rt.add_tvt_split(np.uint([0, 1, 2, 0, 1]))

    def test_add_tvt_split(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_tvt_split(np.uint([0, 1, 2, 0, 1]))
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_cv_split(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_cv_split([0, 1, 2, 3, 4])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_cv_split_bad_splits(self):
        rt = ResultsTable(name="foo", description="a test results table")
        with self.assertRaisesRegex(
            ValueError, "Got non-integer data for cross-validation split"
        ):
            rt.add_cv_split([0.0, 0.1, 0.2, 0.3, 0.4])

    def test_add_true_label(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_true_label([0, 0, 0, 1, 1])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_true_label_str(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_true_label(["a", "a", "a", "b", "b"])
        self.assertIsInstance(rt["true_label"], EnumData)
        np.testing.assert_array_equal(rt.true_label.elements.data, np.array(["a", "b"]))
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_predicted_probability(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_predicted_probability(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]
        )
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_predicted_class(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_predicted_class([0, 0, 1, 1, 1])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_predicted_value(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_predicted_value([0.0, 0.1, 0.2, 0.3, 0.4])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_cluster_label(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_cluster_label([0, 1, 2, 1, 0])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_embedding(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_embedding([[1.1, 2.9], [1.2, 2.8], [1.3, 2.7], [1.4, 2.6], [1.5, 2.5]])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_topk_classes(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_topk_classes([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]])
        with self.get_hdf5io() as io:
            io.write(rt)

    def test_add_topk_probabilities(self):
        rt = ResultsTable(name="foo", description="a test results table")
        rt.add_topk_probabilities(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
        )
        with self.get_hdf5io() as io:
            io.write(rt)
