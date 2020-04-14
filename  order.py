###### test a model with map and show #####
# faster_rcnn
python tools/test.py /home/znjqr/mwh/mmdetection/configs/vhr/faster_rcnn_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_vhr_2/latest.pth --eval mAP
# RetinaNet
python tools/test.py /home/znjqr/mwh/mmdetection/configs/vhr/retinanet_r50_fpn_1x_vhr.py /home/znjqr/mwh/mmdetection/work_dirs/vhr/retinanet_vhr_3/latest.pth --eval mAP
python tools/test.py /home/znjqr/mwh/mmdetection/configs/rsod/retinanet_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/rsod/retinanet_rsod/latest.pth --eval mAP --show
# Cascade_rcnn
python tools/test.py /home/znjqr/mwh/mmdetection/configs/vhr/cascade_rcnn_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/vhr/cascade_rcnn_r50_fpn_1x_vhr/latest.pth --eval mAP --show
# FCOS
python tools/test.py /home/znjqr/mwh/mmdetection/configs/vhr/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x.py /home/znjqr/mwh/mmdetection/work_dirs/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_vhr/latest.pth --eval mAP --show

python tools/test.py /home/znjqr/mwh/mmdetection/configs/rsod/retinanet_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/rsod/retinanet_rsod/latest.pth --eval mAP --show

python tools/test.py /home/znjqr/mwh/mmdetection/configs/vis/retinanet_r50_fpn_1x_vhr.py ./work_dirs/vis/retinanet_3/latest.pth --eval mAP --show

python tools/test.py /home/znjqr/mwh/mmdetection/configs/vis/cascade_rcnn_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/vis/cascade_rcnn/latest.pth --eval mAP --show

python tools/test.py /home/znjqr/mwh/mmdetection/configs/vis/faster_rcnn_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/vis/faster_rcnn/latest.pth --eval mAP --show

python tools/test.py /home/znjqr/mwh/mmdetection/configs/vis/faster_rcnn_r50_fpn_1x.py /home/znjqr/mwh/mmdetection/work_dirs/vis/retinanet/latest.pth --eval mAP --show


# draw loss for loss_cls and loss_bbox
python tools/analyze_logs.py plot_curve ./work_dirs/retinanet_vhr_3/20200315_202258.log.json --keys loss_cls loss_bbox --out retinanet.pdf
python tools/analyze_logs.py plot_curve ./work_dirs/faster_rcnn_r50_fpn_1x_vhr_2/20200316_010557.log.json --keys loss_cls loss_bbox --out ./map/faster_rcnn.pdf
python tools/analyze_logs.py plot_curve /home/znjqr/mwh/mmdetection/work_dirs/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_vhr/20200317_202057.log.json --keys loss_cls loss_bbox loss_centerness --out ./map/fcos_1.pdf

python tools/analyze_logs.py plot_curve /home/znjqr/mwh/mmdetection/work_dirs/vis/retinanet_2/20200326_120250.log.json --keys loss_cls loss_bbox --out ./map/vis1.pdf

python tools/analyze_logs.py plot_curve /home/znjqr/mwh/mmdetection/work_dirs/vis/retinanet_gn+ws/20200406_190025.log.json --keys loss_cls loss_bbox --out ./map/retinanet.pdf

python tools/analyze_logs.py plot_curve /home/znjqr/mwh/mmdetection/work_dirs/vis/cascade_rcnn/20200327_012020.log.json --keys loss --out ./map/vis_casc.pdf