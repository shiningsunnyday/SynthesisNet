python scripts/predict-skeleton.py \
    --skeleton-file results/viz/top_1000/skeletons-top-1000-train.pkl \
    --datasets 0 1 2 3 15 20 22 32 47 60 61 62 64 65 68 72 96 104 156 245 292 295 525 540 812 958 1107 2024 2252 6180 \
    --num_per_class 1 \
    --num_workers 32 \
    --cuda 0 \
    --work-dir /home/msun415/SynTreeNet/results/logs/recognizer/1714155615.1292312
    