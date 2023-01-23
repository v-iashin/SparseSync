CKPT_ID="22-07-28T15-49-45"
python main.py \
    config="./logs/sync_models/$CKPT_ID/cfg-$CKPT_ID.yaml" \
    logging.log_code_state="False" \
    training.finetune="False" \
    training.run_test_only="True" \ 
