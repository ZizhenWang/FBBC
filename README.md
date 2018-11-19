**FBBC** is abbreviation of **F**eature-**B**ased **B**ERT **C**lassifier, and it is easy to adapt to other tasks with few changes.


## How to use

1. Refactor `DataProcessor` to fit your task.
2. run following command as BERT does
    ```
    python simple_classifier.py --do_train=true --do_eval=true\
        --data_dir=YOUR_DATA_PATH --bert_config_file=CONFIG_PATH\
        --vocab_file=VOCAB_PATH --output_dir=OUTPUT_MODEL
    ```
    if you want to fine-tune BERT model, add arg `--finetune=true`
3. if you want to check a model is `frozen` or `finetune`, run 
    ```
    python check_fintuned.py --raw_ckpt=BERT_MODEL_PATH --trained_ckpt=YOUR_MODEL_PATH
    ```

## How it works

- simple_classifier.py
    
    In this code we can choose how to use pre-trained BERT model. Notice there is a `finetune` flag default by `false` which means this classifier uses BERT as feature-based model like `ELMo`. We can also set it to `true` just like the paper does.
    
    Another thing is we simplify the processors here.

- optimizeation.py

    We set trainable variables in this code. We use `bert` as keyword to filter BERT variables out, so you should name your variables without `bert` in it.
