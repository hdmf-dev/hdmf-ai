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

    def add_cv_split(self, data, name="cv_split", description="cross-validation split labels"):
        self.__add_col("CrossValidationSplit", data, name, description)

    def add_true_label(self, data, name="true_label", description='ground truth labels'):
        enum=False
        if isinstance(data[0], (bytes, str)):
            enc = LabelEncoder()
            data = enc.fit_transform(data)
            enum = enc.classes_
        self.__add_col('VectorData', data, name, description, enum=enum)

    def add_predicted_probability(self, data, name="predicted_probability", description="the probability of the predicted class"):
        self.__add_col('ClassProbability', data, name, description)

    def add_predicted_class(self, data, name="predicted_class", description="the predicted class"):
        self.__add_col('ClassLabel', data, name, description)

    def add_predicted_value(self, data, name="predicted_value", description="the predicted regression output"):
        self.__add_col('RegressionOutput', data, name, description)

    def add_cluster_label(self, data, name="cluster_label", description="labels after clustering"):
        self.__add_col('ClusterLabel', data, name, description)

    def add_embedding(self, data, name="embedding", description="dimensionality reduction outputs"):
        self.__add_col('EmbeddedValues', data, name, description)

    def add_topk_classes(self, data, name="topk_classes", description="the top k predicted classes"):
        self.__add_col('TopKClasses', data, name, description)

    def add_topk_probabilities(self, data, name="topk_probabilities", description="the probabilityes of the top k predicted classes"):
        self.__add_col('TopKProbabilities', data, name, description)
