from hdmf.common import register_class, DynamicTable
from hdmf.container import Container


@register_class('ResultsTable', 'hdmf-ml')
class ResultsTable(DynamicTable):


    def __add_col(self, cls, data, name, **kwargs):
        if isinstance(cls, str):
            cls = get_class(cls, 'hdmf-ml')
        return self.add_column(cls(data=data, name=name, **kwargs))

    def add_tvt_split(self, data, name="tvt_split", description=None):
        VectorData = get_class(cls, 'hdmf-common')
        elements = self.__add_col(VectorData(name=f'{name}_elements', data=['train', 'validate', 'test']))
        self.__add_col(data, name, elements=elements)

    def add_cv_split(self, data, name="cv_split", description=None):
        self.__add_col(data, name)

    def add_true_label(self, data, name="true_label", description=None):
        self.__add_col(data, name)

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
