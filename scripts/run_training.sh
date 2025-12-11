python3 src/train_policy.py \
    --train-path data/training/combined/stage1_train.jsonl \
    --eval-path data/training/combined/stage1_eval.jsonl \
    --output-dir models/stage1-context-inference \
    --num-epochs 4 \
    --learning-rate 2e-4 \
    --max-seq-len 1024 \
    --use-4bit

python3 src/train_policy.py \
    --train-path data/training/combined/stage2_train.jsonl \
    --eval-path data/training/combined/stage2_eval.jsonl \
    --output-dir models/stage2-rule-generation \
    --num-epochs 4 \
    --learning-rate 2e-4 \
    --max-seq-len 3072 \
    --use-4bit
