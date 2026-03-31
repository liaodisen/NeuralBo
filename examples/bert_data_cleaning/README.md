# BERT Data Cleaning

Files:
- `bert_data_cleaning.py`: BERT data-cleaning example on TREC.
- `bert_model.py`: BERT classifier and wrapper used by the example.

Run from repo root:
```bash
python examples/bert_data_cleaning/bert_data_cleaning.py --alg AID-KFAC --fine_tune_level 1 --epochs 1000 --batch_size 128 --w_lr 1000
```

Default data path points to `examples/trec/`.
