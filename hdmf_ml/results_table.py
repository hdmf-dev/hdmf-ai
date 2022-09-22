from hdmf.utils import docval, popargs
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import get_class, register_class, DynamicTable, ElementIdentifiers
from hdmf.container import Container
import numpy as np
from sklearn.preprocessing import LabelEncoder

data_type = ('array_data', 'data')

@register_class('ResultsTable', 'hdmf-ml')
class ResultsTable(get_class('ResultsTable', 'hdmf-ml')):

    @docval({'name': 'name',        'type': str,          'default': 'root',
             'doc': 'a name for these results e.g. params1'},
            {'name': 'description', 'type': str,          'default': 'no description',
             'doc': 'a human-readable description of the results e.g. training tricks, parameter, etc.'},
            allow_extra=True)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @docval({'name': 'cls',         'type': (str, type),  'doc': 'data for this column'},
            {'name': 'data',        'type': data_type, 'doc': 'data for this column'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column'},
            allow_extra=True)
    def __add_col(self, **kwargs):
        cls, data, name, description = popargs('cls', 'data', 'name', 'description', kwargs)
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
        return self[name]

    @docval({'name': 'data',        'type': data_type, 'doc': 'data for this column'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'tvt_split'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': 'train/validation/test mask'})
    def add_tvt_split(self, **kwargs):
        """Add mask of 0, 1, 2 indicating which samples were used for training, validation, and testing."""
        kwargs['enum'] = ['train', 'validate', 'test']
        ret = self.__add_col('TrainValidationTestSplit', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'train-validation-test split data'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'tvt_split'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "cross-validation split labels"})
    def add_cv_split(self, **kwargs):
        """Add cross-validation split mask"""
        kwargs['n_splits'] = np.max(kwargs['data']) + 1
        if not isinstance(kwargs['n_splits'], (int, np.integer)):
            raise ValueError('Got non-integer data for cross-validation split')
        return self.__add_col("CrossValidationSplit", **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'ground truth labels for each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'true_label'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': 'ground truth labels'})
    def add_true_label(self, **kwargs):
        """Add ground truth labels for each sample"""
        if isinstance(kwargs['data'][0], (bytes, str)):
            enc = LabelEncoder()
            kwargs['data'] = enc.fit_transform(kwargs['data'])
            kwargs['enum'] = enc.classes_
        return self.__add_col('VectorData', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'probability of sample for each class'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'predicted_probability'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "the probability of the predicted class"})
    def add_predicted_probability(self, **kwargs):
        """Add probability of the sample for each class in the model"""
        return self.__add_col('ClassProbability', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'predicted class lable for each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'predicted_class'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "the predicted class"})
    def add_predicted_class(self, **kwargs):
        """Add predicted class label for each sample"""
        return self.__add_col('ClassLabel', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'predicted value for each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'predicted_value'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "the predicted regression output"})
    def add_predicted_value(self, **kwargs):
        """Add predicted value (i.e. from a regression model) for each sample"""
        return self.__add_col('RegressionOutput', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'cluster label for each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'cluster_label'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "labels after clustering"})
    def add_cluster_label(self, **kwargs):
        """Add cluster label for each sample"""
        return self.__add_col('ClusterLabel', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'embedding of each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'embedding'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "dimensionality reduction outputs"})
    def add_embedding(self, **kwargs):
        """Add embedding (a.k.a. transformation or representation) of each sample"""
        return self.__add_col('EmbeddedValues', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'top-k predicted classes for each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'topk_classes'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "the top k predicted classes"})
    def add_topk_classes(self, **kwargs):
        """Add the top *k* predicted classes for each sample"""
        return self.__add_col('TopKClasses', **kwargs)

    @docval({'name': 'data',        'type': data_type, 'doc': 'probabilities of the top-k predicted classes for each sample'},
            {'name': 'name',        'type': str,          'doc': 'the name of this column', 'default': 'topk_probabilities'},
            {'name': 'description', 'type': str,          'doc': 'a description for this column', 'default': "the probabilityes of the top k predicted classes"})
    def add_topk_probabilities(self, **kwargs):
        """Add probabilities for the top *k* predicted classes for each sample"""
        return self.__add_col('TopKProbabilities', **kwargs)
