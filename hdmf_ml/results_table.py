from hdmf.utils import docval, popargs
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import get_class, register_class, DynamicTable, ElementIdentifiers
from hdmf.container import Container
import numpy as np
from sklearn.preprocessing import LabelEncoder


data_type = ('array_data', 'data')


@register_class('ResultsTable', 'hdmf-ml')
class ResultsTable(get_class('ResultsTable', 'hdmf-ml')):
    # override the auto-generated ResultsTable class

    @docval({'name': 'name',        'type': str,          'default': 'root',
             'doc': 'a name for these results e.g. params1'},
            {'name': 'description', 'type': str,          'default': 'no description',
             'doc': 'a human-readable description of the results e.g. training tricks, parameter, etc.'},
            {'name': 'n_samples', 'type': int, 'doc': 'the number of samples in this table', 'default': None},
            allow_extra=True)
    def __init__(self, **kwargs):
        n_samples = popargs('n_samples', kwargs)
        super().__init__(**kwargs)
        self.__n_samples = n_samples

    @property
    def n_samples(self):
        return self.__n_samples

    @docval({'name': 'cls',         'type': (str, type), 'doc': 'data for this column'},
            {'name': 'data',        'type': data_type,   'doc': 'data for this column'},
            {'name': 'name',        'type': str,         'doc': 'the name of this column'},
            {'name': 'description', 'type': str,         'doc': 'a description for this column'},
            {'name': 'dim2',        'type': str,         'doc': 'the argument holding the second dimension', 'default': None},
            {'name': 'dtype',       'type': type,        'doc': 'the argument holding the second dimension', 'default': None},
            allow_extra=True)
    def __add_col(self, **kwargs):
        """A helper function to handle boiler-plate code for adding columns to a ResultsTable"""
        cls, data, name, description, dim2, dtype = popargs('cls', 'data', 'name', 'description', 'dim2', 'dtype', kwargs)
        if dim2 is not None:
            dim2 = kwargs.pop(dim2)
        if data is None:
            if self.n_samples is None:
                raise ValueError("Must specify n_samples in ResultsTable constructor "
                                 "if you will not be specifying individual column shape")

            shape = (self.n_samples,)
            if dim2 is not None:
                if isinstance(dim2, (int, np.integer)):
                    shape = (self.n_samples, dim2)
                elif isinstance(dim2, (list, tuple)):
                    shape = (self.n_samples, *dim2)
                elif isinstance(dim2, np.array) and len(dim2.shape) == 1:
                    shape = (self.n_samples, *dim2)
                else:
                    ValueError(f"Unrecognized type for dim2: {type(dim2)} - expected integether or 1-D array-like")
            data = H5DataIO(shape=shape, dtype=dtype)

        if name in self:
            raise ValueError(f"Column '{name}' already exists in ResultsTable '{self.name}'")
        if len(self.id) == 0:
            self.id.extend(np.arange(len(data)))
        if len(self.id) != len(data):
            raise ValueError(f'New column {name} of length {len(data)} is not the same length as '
                             f'existings columns of length {len(self.id)}')

        if isinstance(cls, str):
            cls = get_class(cls, 'hdmf-ml')
        self.add_column(data=data, name=name, description=description, col_cls=cls, **kwargs)

        if self.__n_samples is None:
            self.__n_samples = len(data)
        return self[name]

    @docval({'name': 'data',        'type': data_type, 'doc': 'data for this column', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'tvt_split'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': 'train/validation/test mask'})
    def add_tvt_split(self, **kwargs):
        """Add mask of 0, 1, 2 indicating which samples were used for training, validation, and testing."""
        kwargs['enum'] = ['train', 'validate', 'test']
        kwargs['dtype'] = int
        ret = self.__add_col('TrainValidationTestSplit', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'train-validation-test split data', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'tvt_split'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "cross-validation split labels"},
            {'name': 'n_splits',    'type': int,       'doc': 'the number of cross-validation splits', 'default': None})
    def add_cv_split(self, **kwargs):
        """Add cross-validation split mask"""
        if kwargs['data'] is None or isinstance(kwargs['data'], H5DataIO):
            if kwargs['n_splits'] is None:
                raise ValueError("n_splits must be specified if not passing data in")
        else:
            if kwargs['n_splits'] is None:
                kwargs['n_splits'] = np.max(kwargs['data']) + 1
        if not isinstance(kwargs['n_splits'], (int, np.integer)):
            raise ValueError('Got non-integer data for cross-validation split')
        kwargs['dtype'] = int
        return self.__add_col("CrossValidationSplit", **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'ground truth labels for each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'true_label'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': 'ground truth labels'})
    def add_true_label(self, **kwargs):
        """Add ground truth labels for each sample"""
        if isinstance(kwargs['data'][0], (bytes, str)):
            enc = LabelEncoder()
            kwargs['data'] = enc.fit_transform(kwargs['data'])
            kwargs['enum'] = enc.classes_
        kwargs['dtype'] = int
        return self.__add_col('VectorData', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'probability of sample for each class', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'predicted_probability'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "the probability of the predicted class"},
            {'name': 'n_classes',   'type': int,       'doc': 'the number of classes', 'default': None})
    def add_predicted_probability(self, **kwargs):
        """Add probability of the sample for each class in the model"""
        kwargs['dtype'] = float
        kwargs['dim2'] = 'n_classes'
        return self.__add_col('ClassProbability', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'predicted class lable for each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'predicted_class'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "the predicted class"})
    def add_predicted_class(self, **kwargs):
        """Add predicted class label for each sample"""
        kwargs['dtype'] = int
        return self.__add_col('ClassLabel', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'predicted value for each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'predicted_value'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "the predicted regression output"},
            {'name': 'n_dims',      'type': int,       'doc': 'the number of dimensions in the regression output', 'default': None})
    def add_predicted_value(self, **kwargs):
        """Add predicted value (i.e. from a regression model) for each sample"""
        kwargs['dtype'] = float
        kwargs['dim2'] = 'n_dims'
        return self.__add_col('RegressionOutput', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'cluster label for each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'cluster_label'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "labels after clustering"})
    def add_cluster_label(self, **kwargs):
        """Add cluster label for each sample"""
        kwargs['dtype'] = int
        return self.__add_col('ClusterLabel', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'embedding of each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'embedding'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "dimensionality reduction outputs"},
            {'name': 'n_dims',      'type': int,       'doc': 'the number of dimensions in the embedding', 'default': None})
    def add_embedding(self, **kwargs):
        """Add embedding (a.k.a. transformation or representation) of each sample"""
        kwargs['dtype'] = float
        kwargs['dim2'] = 'n_dims'
        return self.__add_col('EmbeddedValues', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'top-k predicted classes for each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'topk_classes'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "the top k predicted classes"},
            {'name': 'k',           'type': int,       'doc': 'the number of top classes', 'default': None})
    def add_topk_classes(self, **kwargs):
        """Add the top *k* predicted classes for each sample"""
        kwargs['dtype'] = int
        kwargs['dim2'] = 'k'
        return self.__add_col('TopKClasses', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'probabilities of the top-k predicted classes for each sample', 'default': None},
            {'name': 'name',        'type': str,       'doc': 'the name of this column', 'default': 'topk_probabilities'},
            {'name': 'description', 'type': str,       'doc': 'a description for this column', 'default': "the probabilities of the top k predicted classes"},
            {'name': 'k',           'type': int,       'doc': 'the number of top classes', 'default': None})
    def add_topk_probabilities(self, **kwargs):
        """Add probabilities for the top *k* predicted classes for each sample"""
        kwargs['dtype'] = float
        kwargs['dim2'] = 'k'
        return self.__add_col('TopKProbabilities', **kwargs)
