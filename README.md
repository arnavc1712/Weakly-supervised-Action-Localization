# Weakly-supervised-Action-Localization

## Download feature files
- [feature_train.npy](https://drive.google.com/uc?export=download&id=15qQIX7EJXmbtZr__U6msnBIzjzAH7ISc)
- [feature_val.npy](https://drive.google.com/uc?export=download&id=1YZcpmHdbiguxNNsZppIj456b3_7Y3a9W)

Store these inside ./data/THUMOS14/

## Train Model
``` python main.py --mode train --epochs 120```

## Evaluation
``` python main.py --mode val --batch_size 1```
