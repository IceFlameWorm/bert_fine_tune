ATEC_CCKS_PATH=/efs/projects/bert_fine_tune/fine_tune/data/train_dev_test/ATEC_CCKS/processed
BERT_BASE_CHINESE_PATH=/efs/downloads/bert/pytorch/bert_base_chinese

python run_classifier_fine_tune.py \
  --task_name atec_ccks \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir  $ATEC_CCKS_PATH \
  --bert_model $BERT_BASE_CHINESE_PATH \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /efs/atec_ccks_fine_tune/
