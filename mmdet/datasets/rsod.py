from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS

from .vhr import VHRDataset


@DATASETS.register_module
class RSODDataset(VHRDataset):

    CLASSES = ('aircraft', 'playground', 'overpass', 'oiltank')

    def __init__(self, **kwargs):
        super(VHRDataset, self).__init__(**kwargs)