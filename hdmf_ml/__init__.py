from hdmf.common import load_namespaces


def __get_resources():
    try:
        from importlib.resources import files
    except ImportError:
        # TODO: Remove when python 3.9 becomes the new minimum
        from importlib_resources import files

    __location_of_this_file = files(__name__)
    __core_ns_file_name = "namespace.yaml"
    __schema_dir = "schema"

    ret = dict()
    ret["namespace_path"] = str(
        __location_of_this_file / __schema_dir / __core_ns_file_name
    )
    return ret


CORE_NAMESPACE = "hdmf-ml"
load_namespaces(__get_resources()["namespace_path"])

from . import results_table
from .results_table import ResultsTable
