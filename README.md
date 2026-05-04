# RotDiffSeg
## Dataset Preparation
The Utilized OVS Datasets (DLRSD, iSAID, Potsdam, Vaihingen)
[BaiduDesk](https://pan.baidu.com/share/init?surl=3D8wUEA_qqrzMc5Z8PCAwg)

(Password:sjtu)

detectron2 is required to install


##How to train our model
python3 train_net.py --config-file configs/vitb_384.yaml

##How to test our model(example)
python3 train_net.py --config-file configs/vitb_384.yaml --eval-only  MODEL.WEIGHTS output/model_0044999.pth OUTPUT_DIR /output/eval

##How to visualize segmentation results
python3 -m cat_seg.visualize_json_results   --input /media/frank/c33f46c2-95b6-4799-a17a-59120b2520a2/VisualOut/eval/inference/sem_seg_predictions.json   --output /media/frank/c33f46c2-95b6-4799-a17a-59120b2520a2/VisualOut/eval/vis   --dataset iSAID_val_sem_seg

