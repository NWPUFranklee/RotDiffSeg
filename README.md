# RotDiffSeg
## Dataset Preparation
The Utilized OVS Datasets (DLRSD, iSAID, Potsdam, Vaihingen)
[BaiduDesk](https://pan.baidu.com/share/init?surl=3D8wUEA_qqrzMc5Z8PCAwg)

(Password:sjtu)

detectron2 is required to install


## How to train our model
python3 train_net.py --config-file configs/vitb_384.yaml

## How to test our model(example)
python3 train_net.py --config-file configs/vitb_384.yaml --eval-only  MODEL.WEIGHTS output/model_0044999.pth OUTPUT_DIR /output/eval

## How to visualize segmentation results
python3 -m cat_seg.visualize_json_results   --input /media/frank/c33f46c2-95b6-4799-a17a-59120b2520a2/VisualOut/eval/inference/sem_seg_predictions.json   --output /media/frank/c33f46c2-95b6-4799-a17a-59120b2520a2/VisualOut/eval/vis   --dataset iSAID_val_sem_seg

 ```bibtex
@ARTICLE{10962188,
 author={Cao, Qinglong and Chen, Yuntian and Ma, Chao and Yang, Xiaokang},
 journal={IEEE Transactions on Geoscience and Remote Sensing}, 
 title={Open-Vocabulary High-Resolution Remote Sensing Image Semantic Segmentation}, 
 year={2025},
 volume={63},
 number={},
 pages={1-14},
 keywords={Remote sensing;Semantics;Semantic segmentation;Transformers;Adaptation models;Feature extraction;Convolutional neural networks;Computational modeling;Accuracy;Training;Open-vocabulary semantic segmentation;remote sensing image segmentation;scale variation;varying orientations},
 doi={10.1109/TGRS.2025.3559557}}

@ARTICLE{11215798,
 author={Zhang, Xiaokang and Zhou, Chufeng and Huang, Jianzhong and Zhang, Lefei},
 journal={IEEE Transactions on Geoscience and Remote Sensing}, 
 title={TPOV-Seg: Textually Enhanced Prompt Tuning of Vision-Language Models for Open-Vocabulary Remote Sensing Semantic Segmentation}, 
 year={2025},
 volume={63},
 number={},
 pages={1-17},
 keywords={Remote sensing;Semantic segmentation;Adaptation models;Semantics;Land surface;Training;Tuning;Transformers;Vegetation mapping;Visualization;Open vocabulary;prompt tuning;remote sensing;semantic segmentation;zero shot},
 doi={10.1109/TGRS.2025.3624767}}
