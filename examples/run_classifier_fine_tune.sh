ATEC_CCKS_PATH=/efs/projects/bert_fine_tune/fine_tune/data/train_dev_test/ATEC_CCKS/processed
#BERT_MODEL_PATH=/efs/downloads/bert/pytorch/bert_base_chinese
BERT_MODEL_PATH=/efs/fine_tune/atec_ccks_fine_tune_3/

python run_classifier_fine_tune.py \
  --task_name atec_ccks \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir  $ATEC_CCKS_PATH \
  --bert_model $BERT_MODEL_PATH \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --output_dir /efs/fine_tune/atec_ccks_fine_tune_4/
