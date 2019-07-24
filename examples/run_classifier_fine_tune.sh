DATA_DIR=/efs/projects/bert_fine_tune/fine_tune/data/train_dev_test/LCQMC/processed
BERT_MODEL_PATH=/efs/downloads/bert/pytorch/bert_base_chinese
#BERT_MODEL_PATH=/efs/fine_tune/lcqmc/pointwise/lcqmc_fine_tune_40_1_5e-6/
#BERT_MODEL_PATH=/efs/fine_tune/lcqmc/pointwise/lcqmc_fine_tune_sample/

python run_classifier_pointwise.py \
  --task_name lcqmc \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir  $DATA_DIR \
  --bert_model $BERT_MODEL_PATH \
  --max_seq_length 40 \
  --train_batch_size 32 \
  --learning_rate 8e-6 \
  --num_train_epochs 1.0 \
  --output_dir /efs/fine_tune/lcqmc/pointwise/lcqmc_fine_tune_40_1_8e-6/
