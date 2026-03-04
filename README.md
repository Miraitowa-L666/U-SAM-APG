# U-SAM-APG

**Adaptive Box Prompt Generation for U-SAM: Fully Automatic Rectal Cancer CT Segmentation Without Manual Annotation**

Yuhan Liu  
Hebei University of Science and Technology  
Shijiazhuang, China  
Email: yyyh0213@163.com


## Model

We provide our implementation of U-SAM-APG, an improved U-SAM framework with Adaptive Prompt Generator (APG) for fully automatic rectal cancer CT segmentation.

- Core module: [models/adaptive_prompt_generator.py](models/adaptive_prompt_generator.py)
- Training scripts: [scripts/](scripts/)
- Dataset loader: [dataset/rectum_dataloader.py](dataset/rectum_dataloader.py)

## Datasets

Experiments were conducted on the public **CARE** dataset (DataV6 version) [2]. The dataset contains abdominal enhanced CT scans from 398 pathologically confirmed primary rectal adenocarcinoma patients, with pixel-level masks annotated by clinical experts.

**Download:**
- Paper: https://www.nature.com/articles/s43856-025-00953-0
- Dataset: https://drive.google.com/file/d/1X_JTfD8Ch-IxmG5VHtKk_xGZT336Fl1Q/view?usp=drive_link

**License:** CC BY-NC 4.0

**IRB Approval:** First Affiliated Hospital of Anhui Medical University (No. Quick-PJ 2023-13-34)

**Dataset Structure:**
```
/path/to/CARE/dataset/
├── train/
│   ├── train_bbox.csv
│   └── train_npz/
└── test/
    ├── test_bbox.csv
    └── test_npz/
```

## Get Started

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA compatible GPU (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Pre-trained Weights

**SAM ViT-B weights:** Download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth and place in `checkpoints/`.

**U-SAM-APG checkpoints:** Provided in `checkpoints/` directory:
- `stage1_best_0.7748.pth` - APG pre-training checkpoint
- `stage2_best_0.6984.pth` - End-to-end fine-tuning checkpoint

## Training

The training follows a two-stage strategy as described in the paper.

### Stage 1: APG Pre-training

```bash
python scripts/train_stage1.py \
  --dataset rectum \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-3 \
  --output_dir ./results/stage1_training \
  --root /path/to/CARE/dataset
```

### Stage 2: End-to-End Fine-tuning

```bash
python scripts/train_stage2.py \
  --dataset rectum \
  --use_adaptive_prompt \
  --adaptive_prompt_type box \
  --use_attention \
  --heatmap_source foreground \
  --heatmap_loss_weight 0.15 \
  --rectum_heatmap_loss_weight 0.15 \
  --classification_pretrained ./checkpoints/stage1_best_0.7748.pth \
  --freeze_classification_epochs 3 \
  --epochs 20 \
  --batch_size 8 \
  --lr 5e-4 \
  --lr_vit 5e-4 \
  --lr_backbone 5e-4 \
  --lr_classification 1e-5 \
  --warmup \
  --lr_schedule cosine \
  --warmup_ratio 0.05 \
  --dice_weight 0.6 \
  --box_threshold 0.3 \
  --box_margin_ratio 0.15 \
  --output_dir ./results/stage2_training \
  --root /path/to/CARE/dataset
```



## Evaluation

```bash
python scripts/evaluate.py \
  --dataset rectum \
  --use_adaptive_prompt \
  --adaptive_prompt_type box \
  --eval \
  --resume checkpoints/stage2_best_0.6984.pth \
  --root /path/to/CARE/dataset
```





## Contact

For questions, please contact: Yuhan Liu (yyyh0213@163.com)

School of Information Science and Engineering  
Hebei University of Science and Technology

## Acknowledgment

This work is based on [U-SAM](https://github.com/kanydao/U-SAM) and [Segment Anything](https://github.com/facebookresearch/segment-anything). We thank the authors for releasing their code.

## Citation

If this code is helpful for your study, please cite:

```
@article{liu2026adaptive,
  title={Adaptive Box Prompt Generation for U-SAM: Fully Automatic Rectal Cancer CT Segmentation Without Manual Annotation},
  author={Liu, Yuhan},
  journal={Under Review},
  year={2026}
}
```
