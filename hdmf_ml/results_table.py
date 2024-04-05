from hdmf.utils import docval, popargs, get_docval
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import get_class, register_class, VectorData, EnumData, DynamicTable, DynamicTableRegion
import numpy as np
from sklearn.preprocessing import LabelEncoder


data_type = ("array_data", "data")

SupervisedOutput = get_class("SupervisedOutput", "hdmf-ml")
TrainValidationTestSplit = get_class("TrainValidationTestSplit", "hdmf-ml")
CrossValidationSplit = get_class("CrossValidationSplit", "hdmf-ml")
ClassProbability = get_class("ClassProbability", "hdmf-ml")
ClassLabel = get_class("ClassLabel", "hdmf-ml")
TopKProbabilities = get_class("TopKProbabilities", "hdmf-ml")
TopKClasses = get_class("TopKClasses", "hdmf-ml")
RegressionOutput = get_class("RegressionOutput", "hdmf-ml")
ClusterLabel = get_class("ClusterLabel", "hdmf-ml")
EmbeddedValues = get_class("EmbeddedValues", "hdmf-ml")

_AutoGenResultsTable = get_class("ResultsTable", "hdmf-ml")


@register_class("ResultsTable", "hdmf-ml")
class ResultsTable(_AutoGenResultsTable):
    # extend the auto-generated ResultsTable class

    @docval(
        {
            "name": "name",
            "type": str,
            "default": "root",
            "doc": "a name for these results e.g. params1",
        },
        {
            "name": "description",
            "type": str,
            "default": "no description",
            "doc": "a human-readable description of the results e.g. training tricks, parameter, etc.",
        },
        {
            "name": "n_samples",
            "type": int,
            "doc": "the number of samples in this table",
            "default": None,
        },
        *get_docval(_AutoGenResultsTable.__init__, "id", "columns", "colnames", "target_tables"),
        allow_extra=True,
    )
    def __init__(self, **kwargs):
        n_samples = popargs("n_samples", kwargs)
        super().__init__(**kwargs)
        self.__n_samples = n_samples

    @property
    def n_samples(self):
        return self.__n_samples

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "row indices of the samples used in the machine learning algorithm",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": None,
        },
        {
            "name": "table",
            "type": DynamicTable,
            "doc": "the referenced table",
            "default": None,
        }
    )
    def add_samples(self, **kwargs):
        data, description, table = popargs("data", "description", "table", kwargs)
        self.add_column(name="samples", data=data, description=description, col_cls=DynamicTableRegion, table=table, **kwargs)

        if self.__n_samples is None:
            self.__n_samples = len(data)
        return self["samples"]


    @docval(
        {"name": "col_cls", "type": type, "doc": "class for this column"},
        {"name": "data", "type": data_type, "doc": "data for this column"},
        {"name": "name", "type": str, "doc": "the name of this column"},
        {"name": "description", "type": str, "doc": "a description for this column"},
        {
            "name": "dim2_kwarg",
            "type": str,
            "doc": (
                "the name of the argument in kwargs holding the size of the other dimension(s)"
                "as an int for a 2D shape or a list/tuple/1-D array for an N-D shape where "
                "N is equal to the length of the list/tuple/1-D array + 1"
            ),
            "default": None,
        },
        {
            "name": "dtype",
            "type": type,
            "doc": "the dtype for the column data",
            "default": None,
        },
        allow_extra=True,
    )
    def __add_col(self, **kwargs):
        """A helper function to handle boiler-plate code for adding columns to a ResultsTable"""
        col_cls, data, name, description, dim2_kwarg, dtype = popargs(
            "col_cls", "data", "name", "description", "dim2_kwarg", "dtype", kwargs
        )
        # get the size of the other dimension(s) from kwargs
        if dim2_kwarg is not None:
            dim2 = kwargs.pop(dim2_kwarg)
        if data is None:
            if self.n_samples is None:
                raise ValueError(
                    "Must specify n_samples in ResultsTable constructor "
                    "if you will not be specifying individual column shape"
                )

            shape = (self.n_samples,)
            if dim2 is not None:
                if isinstance(dim2, (int, np.integer)):
                    # dim2 is an integer, so column is 2D
                    shape = (self.n_samples, dim2)
                elif isinstance(dim2, (list, tuple)):
                    # dim2 is a list or tuple, so shape is N-D
                    shape = (self.n_samples, *dim2)
                elif isinstance(dim2, np.array) and len(dim2.shape) == 1:
                    # dim2 is a 1D array, so shape is N-D
                    shape = (self.n_samples, *dim2)
                else:
                    ValueError(f"Unrecognized type for dim2: {type(dim2)} - expected integer or 1-D array-like")

            # create empty DataIO object
            data = H5DataIO(shape=shape, dtype=dtype)

        if name in self:
            raise ValueError(f"Column '{name}' already exists in ResultsTable '{self.name}'")
        if len(self.id) == 0:
            self.id.extend(np.arange(len(data)))
        elif len(self.id) != len(data):
            raise ValueError(
                f"New column {name} of length {len(data)} is not the same length as "
                f"existings columns of length {len(self.id)}"
            )

        self.add_column(data=data, name=name, description=description, col_cls=col_cls, **kwargs)

        if self.__n_samples is None:
            self.__n_samples = len(data)
        return self[name]

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "train/validation/test mask (enum: train, validation, test) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "train/validation/test mask",
        },
    )
    def add_tvt_split(self, **kwargs):
        """Add mask of 0, 1, 2 indicating which samples were used for training, validation, and testing."""
        kwargs["name"] = "tvt_split"
        kwargs["enum"] = ["train", "validate", "test"]
        kwargs["dtype"] = int
        return self.__add_col(TrainValidationTestSplit, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "cross-validation split labels (int) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "cross-validation split labels",
        },
        {
            "name": "n_splits",
            "type": int,
            "doc": "the number of cross-validation splits",
            "default": None,
        },
    )
    def add_cv_split(self, **kwargs):
        """Add cross-validation split mask"""
        kwargs["name"] = "cv_split"
        if kwargs["data"] is None or isinstance(kwargs["data"], H5DataIO):
            if kwargs["n_splits"] is None:
                raise ValueError("n_splits must be specified if not passing data in")
        else:
            if kwargs["n_splits"] is None:
                # set n_splits to one more than the max value of the data
                kwargs["n_splits"] = np.max(kwargs["data"]) + 1
        if not isinstance(kwargs["n_splits"], (int, np.integer)):
            # this should have been checked in docval?
            raise ValueError("Got non-integer data for cross-validation split")
        kwargs["dtype"] = int
        return self.__add_col(CrossValidationSplit, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "ground truth labels (int, bytes, or str) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "ground truth labels",
        },
    )
    def add_true_label(self, **kwargs):
        """Add ground truth labels (int, bytes, or str) for each sample"""
        kwargs["name"] = "true_label"
        if isinstance(kwargs["data"][0], (bytes, str)):
            # if data are strings, convert to enum data type (data are ints, enum elements are strings)
            enc = LabelEncoder()
            kwargs["data"] = np.uint(enc.fit_transform(kwargs["data"]))
            kwargs["dtype"] = np.uint
            kwargs["enum"] = enc.classes_
            return self.__add_col(EnumData, **kwargs)
        kwargs["dtype"] = int
        return self.__add_col(VectorData, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "probability of sample (float) for each class",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "the probability of the predicted class",
        },
        {
            "name": "n_classes",
            "type": int,
            "doc": "the number of classes, used to define the shape of the column only if data is None",
            "default": None,
        },
    )
    def add_predicted_probability(self, **kwargs):
        """Add probability of the sample for each class in the model"""
        kwargs["name"] = "predicted_probability"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_classes"
        # n_classes kwarg is passed into __add_col and will be read as the length of the second dimension
        # of the data only if the data kwarg is None.
        return self.__add_col(ClassProbability, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "predicted class label (int) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "the predicted class",
        },
    )
    def add_predicted_class(self, **kwargs):
        """Add predicted class label (int) for each sample"""
        kwargs["name"] = "predicted_class"
        kwargs["dtype"] = np.uint
        return self.__add_col(ClassLabel, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "predicted value (float) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "the predicted regression output",
        },
        {
            "name": "n_dims",
            "type": int,
            "doc": (
                "the number of dimensions in the regression output, "
                "used to define the shape of the column only if data is None"
            ),
            "default": None,
        },
    )
    def add_predicted_value(self, **kwargs):
        """Add predicted value (i.e. from a regression model) for each sample"""
        kwargs["name"] = "predicted_value"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_dims"
        # n_dims kwarg is passed into __add_col and will be read as the length of the second dimension
        # of the data only if the data kwarg is None.
        return self.__add_col(RegressionOutput, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "cluster label (int) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "labels after clustering",
        },
    )
    def add_cluster_label(self, **kwargs):
        """Add cluster label for each sample"""
        kwargs["name"] = "cluster_label"
        kwargs["dtype"] = int
        return self.__add_col(ClusterLabel, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "embedding (float) of each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "dimensionality reduction outputs",
        },
        {
            "name": "n_dims",
            "type": int,
            "doc": (
                "the number of dimensions in the embedding, "
                "used to define the shape of the column only if data is None"
            ),
            "default": None,
        },
    )
    def add_embedding(self, **kwargs):
        """Add embedding (a.k.a. transformation or representation) of each sample"""
        kwargs["name"] = "embedding"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_dims"
        # n_dims kwarg is passed into __add_col and will be read as the length of the second dimension
        # of the data only if the data kwarg is None.
        return self.__add_col(EmbeddedValues, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "top-k predicted classes (int) for each sample",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "the top k predicted classes",
        },
        {
            "name": "k",
            "type": int,
            "doc": "the number of top classes, used to define the shape of the column only if data is None",
            "default": None,
        },
    )
    def add_topk_classes(self, **kwargs):
        """Add the top *k* predicted classes for each sample"""
        kwargs["name"] = "topk_classes"
        kwargs["dtype"] = int
        kwargs["dim2_kwarg"] = "k"
        # k kwarg is passed into __add_col and will be read as the length of the second dimension
        # of the data only if the data kwarg is None.
        return self.__add_col(TopKClasses, **kwargs)

    @docval(
        {
            "name": "data",
            "type": data_type,
            "doc": "probabilities (float) of the top-k predicted classes for each sample",
            "default": None,
        },
        {
            "name": "name",
            "type": str,
            "doc": "the name of this column",
            "default": "topk_probabilities",
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "the probabilities of the top k predicted classes",
        },
        {
            "name": "k",
            "type": int,
            "doc": "the number of top predicted classes, used to define the shape of the column only if data is None",
            "default": None,
        },
    )
    def add_topk_probabilities(self, **kwargs):
        """Add probabilities for the top *k* predicted classes for each sample"""
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "k"
        # k kwarg is passed into __add_col and will be read as the length of the second dimension
        # of the data only if the data kwarg is None.
        return self.__add_col(TopKProbabilities, **kwargs)
