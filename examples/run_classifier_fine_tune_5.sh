DATA_SET=LCQMC
TASK_NAME=$(echo ${DATA_SET} | tr '[A-Z]' '[a-z]')
MODEL_TYPE=pairwise2
LEARNING_RATE=2e-5
EPOCHS=1.0
EPOCH_SPAN=1
MAX_SEQ_LENGTH=40

DATA_DIR=/efs/projects/bert_fine_tune/fine_tune/data/train_dev_test/${DATA_SET}/processed
BERT_MODEL_PATH=/efs/downloads/bert/pytorch/bert_base_chinese
#BERT_MODEL_PATH=/efs/fine_tune/${TASK_NAME}/${MODEL_TYPE}/${TASK_NAME}_fine_tune_${MAX_SEQ_LENGTH}_1_${LEARNING_RATE}/
#BERT_MODEL_PATH=/efs/fine_tune/lcqmc/pointwise/lcqmc_fine_tune_sample/


python run_classifier_${MODEL_TYPE}.py \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir  ${DATA_DIR} \
  --bert_model ${BERT_MODEL_PATH} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --train_batch_size 32 \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs ${EPOCHS} \
  --output_dir /efs/fine_tune/${TASK_NAME}/${MODEL_TYPE}/${TASK_NAME}_fine_tune_${MAX_SEQ_LENGTH}_${EPOCH_SPAN}_${LEARNING_RATE}/
