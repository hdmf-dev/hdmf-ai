from hdmf.common import get_class, register_class, DynamicTable
from hdmf.container import Container
import numpy as np
from sklearn.preprocessing import LabelEncoder


@register_class('ResultsTable', 'hdmf-ml')
class ResultsTable(get_class('ResultsTable', 'hdmf-ml')):

    def __add_col(self, cls, data, name, description, **kwargs):
        if len(self.id) < len(data):
            self.id.extend(np.arange(len(self.id), len(data)))
        if isinstance(cls, str):
            cls = get_class(cls, 'hdmf-ml')
        self.add_column(data=data, name=name, description=description, **kwargs)

    def add_tvt_split(self, data, name="tvt_split", description='train/validation/test mask'):
        """Add mask of 0, 1, 2 indicating which samples were used for trainingi validation, and testing."""
        ret = self.__add_col('TrainValidationTestSplit', data, name, description,
                             enum=['train', 'validate', 'test'])

    def add_cv_split(self, data, name="cv_split", description=None):
        self.__add_col(data, name)

    def add_true_label(self, data, name="true_label", description='ground truth labels'):
        enum=False
        if isinstance(data[0], str):
            enc = LabelEncoder()
            data = enc.fit_transform(data)
            enum = enc.classes_
        self.__add_col('VectorData', data, name, description, enum=enum)

    def add_predicted_probability(self, data, name="predicted_probability", description=None):
        self.__add_col(data, name)

    def add_predicted_class(self, data, name="predicted_class", description=None):
        self.__add_col(data, name)

    def add_predicted_value(self, data, name="predicted_value", description=None):
        self.__add_col(data, name)

    def add_cluster_label(self, data, name="cluster_label", description=None):
        self.__add_col(data, name)

    def add_embedding(self, data, name="embedding", description=None):
        self.__add_col(data, name)

    def add_topk_classes(self, data, name="topk_classes", description=None):
        self.__add_col(data, name)

    def add_topk_probabilities(self, data, name="topk_probabilities", description=None):
        self.__add_col(data, name)
