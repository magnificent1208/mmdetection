from mmdet.core import eval_map, eval_recalls

from .pipelines import Compose
from .registry import DATASETS
from .vhr import VHRDataset
from .custom import CustomDataset


@DATASETS.register_module
class RSODDataset(VHRDataset):

    CLASSES = ('aircraft', 'playground', 'overpass', 'oiltank')

    def __init__(self, **kwargs):
        super(VHRDataset, self).__init__(**kwargs)