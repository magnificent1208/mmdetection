import subprocess
from mmcv import Config


work_dir = None
a = 'python tools/test.py configs/vhr/cascade_rcnn_r50_fpn_1x.py work_dirs/vhr/cascade_rcnn_r50_fpn_1x_vhr/latest.pth --eval mAP'
b = 'python /home/user1/mwh/mmdetection/tools/test.py'

if __name__ == '__main__':
    output = subprocess.Popen(a, shell=True)
    output.wait()


