from hdmf.utils import docval, popargs, get_docval
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import get_class, register_class, VectorData, EnumData, DynamicTable, DynamicTableRegion
import numpy as np
from sklearn.preprocessing import LabelEncoder


# NOTE: these classes are generated based on the schema
SupervisedOutput = get_class("SupervisedOutput", "hdmf-ai")
TrainValidateTestSplit = get_class("TrainValidateTestSplit", "hdmf-ai")
CrossValidationSplit = get_class("CrossValidationSplit", "hdmf-ai")
ClassProbability = get_class("ClassProbability", "hdmf-ai")
ClassLabel = get_class("ClassLabel", "hdmf-ai")
TopKProbabilities = get_class("TopKProbabilities", "hdmf-ai")
TopKClasses = get_class("TopKClasses", "hdmf-ai")
RegressionOutput = get_class("RegressionOutput", "hdmf-ai")
ClusterLabel = get_class("ClusterLabel", "hdmf-ai")
EmbeddedValues = get_class("EmbeddedValues", "hdmf-ai")

_AutoGenResultsTable = get_class("ResultsTable", "hdmf-ai")


@register_class("ResultsTable", "hdmf-ai")
class ResultsTable(_AutoGenResultsTable):
    """A table for storing results of AI / machine learning algorithms. This table is designed to store
    the results of a machine learning algorithm, including the predicted class or value, the probability
    of the predicted class, the true class or value, and the embedding of the sample. This table can also
    store the results of a clustering algorithm, including the cluster label of the sample. The table
    can also store the results of a cross-validation algorithm, including the split label of the sample.
    """
    # NOTE this extends the auto-generated ResultsTable class

    @docval(
        {
            "name": "name",
            "type": str,
            "default": "root",
            "doc": "A name for the results table.",
        },
        {
            "name": "description",
            "type": str,
            "default": "no description",
            "doc": "A human-readable description of the results, e.g. training tricks, parameters, etc.",
        },
        {
            "name": "n_samples",
            "type": int,
            "doc": "The number of samples in this table.",
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
            "type": ("array_data", "data"),
            "doc": "Row indices of the samples used in the AI / machine learning algorithm.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": (
                "A selection of rows from another DynamicTable that represent the input to the AI/ML algorithm."
            ),
        },
        {
            "name": "table",
            "type": DynamicTable,
            "doc": "The referenced input table.",
            "default": None,
        }
    )
    def add_samples(self, **kwargs):
        data, description, table = popargs("data", "description", "table", kwargs)
        self.add_column(
            name="samples",
            data=data,
            description=description,
            col_cls=DynamicTableRegion,
            table=table,
            **kwargs
        )

        if self.__n_samples is None:
            self.__n_samples = len(data)
        return self["samples"]


    @docval(
        {"name": "col_cls", "type": type, "doc": "Class for this column."},
        {"name": "data", "type": ("array_data", "data"), "doc": "Data for this column."},
        {"name": "name", "type": str, "doc": "Name of this column."},
        {"name": "description", "type": str, "doc": "Description for this column."},
        {
            "name": "dim2_kwarg",
            "type": str,
            "doc": (
                "The name of the argument in `kwargs` holding the size of the other dimension(s)"
                "as an int for a 2D shape or a list/tuple/1-D array for an N-D shape where "
                "N is equal to the length of the list/tuple/1-D array + 1."
            ),
            "default": None,
        },
        {
            "name": "dtype",
            "type": type,
            "doc": "The dtype for the column data.",
            "default": None,
        },
        allow_extra=True,
    )
    def __add_col(self, **kwargs):
        """A helper function to handle boiler-plate code for adding columns to a ResultsTable."""
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
                    "if you will not be specifying individual column shape."
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
            "type": ("array_data", "data"),
            "doc": "Train/validate/test mask (enum: 'train', 'validate', 'test') for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to indicate if a sample was used for training, validation, or testing.",
        },
    )
    def add_tvt_split(self, **kwargs):
        """Add mask of 0, 1, 2 indicating which samples were used for training, validation, and testing.

        Input values should be 0, 1, or 2, corresponding to 'train', 'validate', and 'test' respectively.
        """
        kwargs["name"] = "tvt_split"
        kwargs["enum"] = ["train", "validate", "test"]
        kwargs["dtype"] = int
        return self.__add_col(TrainValidateTestSplit, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Cross-validation split labels (int) for each sample, starting from 0.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column",
            "default": "A column for storing which cross validation split a sample is part of, starting from 0.",
        },
        {
            "name": "n_splits",
            "type": int,
            "doc": "The number of cross-validation splits.",
            "default": None,
        },
    )
    def add_cv_split(self, **kwargs):
        """Add cross-validation split mask"""
        kwargs["name"] = "cv_split"
        if kwargs["data"] is None or isinstance(kwargs["data"], H5DataIO):
            if kwargs["n_splits"] is None:
                raise ValueError("n_splits must be specified if not passing data in.")
        else:
            if kwargs["n_splits"] is None:
                # set n_splits to one more than the max value of the data
                kwargs["n_splits"] = np.max(kwargs["data"]) + 1
                if not isinstance(kwargs["n_splits"], (int, np.integer)):
                    raise ValueError("Got non-integer data for cross-validation split")
        kwargs["dtype"] = int
        return self.__add_col(CrossValidationSplit, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Ground truth labels (int, bytes, or str) for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": (
                "A column to store the true labels for each sample."
                "The `training_labels` attribute on other columns in the ResultsTable should reference this column, "
                "if present."
            ),
        },
    )
    def add_true_label(self, **kwargs):
        """Add ground truth labels (int, bytes, or str) for each sample.

        If the labels are bytes or str, they will be converted to integers and the column will be an EnumData column.
        """
        kwargs["name"] = "true_label"
        if isinstance(kwargs["data"][0], (bytes, str)):
            # if data are strings, convert to enum data type (data are ints, enum elements are strings)
            enc = LabelEncoder()
            kwargs["data"] = np.uint(enc.fit_transform(kwargs["data"]))
            kwargs["dtype"] = np.uint
            kwargs["enum"] = enc.classes_
            return self.__add_col(EnumData, **kwargs)
        else:
            kwargs["dtype"] = int
            return self.__add_col(VectorData, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Probability of sample (float) for each class.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "A column to store the class probability for each class across the samples.",
        },
        {
            "name": "n_classes",
            "type": int,
            "doc": "the number of classes, used to define the shape of the column only if data is None",
            "default": None,
        },
    )
    def add_predicted_probability(self, **kwargs):
        """Add probability of the sample for each class in the model."""
        kwargs["name"] = "predicted_probability"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_classes"
        # `n_classes` kwarg is passed into `__add_col` and will be read as the length of the second dimension
        # of the data only if the `data` kwarg is None.
        return self.__add_col(ClassProbability, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Predicted class label (int) for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "a description for this column",
            "default": "A column to store which class a sample was classified as.",
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
            "type": ("array_data", "data"),
            "doc": "Probabilities (float) of the top-k predicted classes for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to store the probabilities for the top k predicted classes.",
        },
        {
            "name": "k",
            "type": int,
            "doc": "The number of top predicted classes, used to define the shape of the column only if data is None.",
            "default": None,
        },
    )
    def add_topk_probabilities(self, **kwargs):
        """Add probabilities for the top *k* predicted classes for each sample."""
        kwargs["name"] = "topk_probabilities"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "k"
        # `k` kwarg is passed into `__add_col` and will be read as the length of the second dimension
        # of the data only if the `data` kwarg is None.
        return self.__add_col(TopKProbabilities, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Top-k predicted classes (int) for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to store the top k classes.",
        },
        {
            "name": "k",
            "type": int,
            "doc": "The number of top classes, used to define the shape of the column only if data is None",
            "default": None,
        },
    )
    def add_topk_classes(self, **kwargs):
        """Add the top *k* predicted classes for each sample."""
        kwargs["name"] = "topk_classes"
        kwargs["dtype"] = int
        kwargs["dim2_kwarg"] = "k"
        # `k` kwarg is passed into `__add_col` and will be read as the length of the second dimension
        # of the data only if the `data` kwarg is None.
        return self.__add_col(TopKClasses, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Predicted value (float) for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to store regression outputs for each sample.",
        },
        {
            "name": "n_dims",
            "type": int,
            "doc": (
                "The number of dimensions in the regression output, "
                "used to define the shape of the column only if data is None"
            ),
            "default": None,
        },
    )
    def add_predicted_value(self, **kwargs):
        """Add predicted value (i.e. from a regression model) for each sample."""
        kwargs["name"] = "predicted_value"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_dims"
        # `n_dims` kwarg is passed into `__add_col` and will be read as the length of the second dimension
        # of the data only if the `data` kwarg is None.
        return self.__add_col(RegressionOutput, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Cluster label (int) for each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to store which cluster a sample was clustered into.",
        },
    )
    def add_cluster_label(self, **kwargs):
        """Add cluster label for each sample."""
        kwargs["name"] = "cluster_label"
        kwargs["dtype"] = int
        return self.__add_col(ClusterLabel, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Embedding (float) of each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to store embeddings, e.g., from dimensionality reduction, for each sample.",
        },
        {
            "name": "n_dims",
            "type": int,
            "doc": (
                "The number of dimensions in the embedding, "
                "used to define the shape of the column only if data is None"
            ),
            "default": None,
        },
    )
    def add_embedding(self, **kwargs):
        """Add embedding (a.k.a. transformation or representation) of each sample."""
        kwargs["name"] = "embedding"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_dims"
        # `n_dims` kwarg is passed into `__add_col` and will be read as the length of the second dimension
        # of the data only if the `data` kwarg is None.
        return self.__add_col(EmbeddedValues, **kwargs)

    @docval(
        {
            "name": "data",
            "type": ("array_data", "data"),
            "doc": "Embedding (float) of each sample.",
            "default": None,
        },
        {
            "name": "description",
            "type": str,
            "doc": "A description for this column.",
            "default": "A column to store embeddings, e.g., from dimensionality reduction, for each sample.",
        },
        {
            "name": "n_dims",
            "type": int,
            "doc": (
                "The number of dimensions in the embedding, "
                "used to define the shape of the column only if data is None"
            ),
            "default": None,
        },
    )
    def add_viz_embedding(self, **kwargs):
        """Add embedding (a.k.a. transformation or representation) of each sample."""
        kwargs["name"] = "viz_embedding"
        kwargs["dtype"] = float
        kwargs["dim2_kwarg"] = "n_dims"
        # `n_dims` kwarg is passed into `__add_col` and will be read as the length of the second dimension
        # of the data only if the `data` kwarg is None.
        return self.__add_col(EmbeddedValues, **kwargs)
