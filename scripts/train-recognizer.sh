# python scripts/predict-skeleton.py \
#     --skeleton-file results/viz/top_1000/skeletons-top-1000-train.pkl \
#     --datasets 0 1 2 3 15 20 22 32 47 60 61 62 64 65 68 72 96 104 156 245 292 295 525 540 812 958 1107 2024 2252 6180 \
#     --num_per_class 1 \
#     --num_workers 32 \
#     --cuda 0 \
#     --work-dir /home/msun415/SynTreeNet/results/logs/recognizer/1714155615.1292312
    
python scripts/predict-skeleton.py \
    --skeleton-file results/viz/skeletons-train.pkl \
    --datasets 0 1 2 3 5 6 8 9 10 13 14 15 16 17 18 20 23 25 28 31 36 39 57 76 84 103 118 119 155 217 \
    --num_per_class 1 \
    --num_workers 32 \
    --cuda 1 \
    --top_k 4 \
    --max_vis_per_class 50 \
    --vis_class_criteria size_small \
    --work-dir /home/msun415/SynTreeNet/results/logs/recognizer/1714500818.0444856 \
    --ckpt /home/msun415/SynTreeNet/results/logs/recognizer/1714500818.0444856/ckpts.epoch=1-val_loss=0.14.ckpt