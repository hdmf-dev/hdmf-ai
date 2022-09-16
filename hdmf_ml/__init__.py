from hdmf.common import load_namespaces, get_class

def __get_resources():
    from pkg_resources import resource_filename
    from os.path import join
    __core_ns_file_name = 'namespace.yaml'

    ret = dict()
    ret['namespace_path'] = join(resource_filename(__name__, '../ml'), __core_ns_file_name)
    return ret


CORE_NAMESPACE = 'hdmf-ml'
load_namespaces(__get_resources()['namespace_path'])

from . import results_table

from .results_table import ResultsTable
