# Regression
Abstract: this repo includes a pipeline using tf.keras for the anonymised data regression. Moreover, weights and trained model are provided. I have not included EDA here because it's simple .csv file, it's needed only to be normalized (done using layer). 

[Data & Predictions](https://drive.google.com/drive/folders/1SCa-A6rMtelU_3UzplbZGn0kXQJny0lx?usp=sharing)

## Files Description 
Task 1-2.ipynb contains colab notebook as solution for fast sum() and islands count.

Task 3.ipynb is main file for regression. Train_save_preds.py doing same (train model and saves predictions).

## Training Results
| Architecture | RMSE | Epochs | Steps_per_epoch | Loss function | Optimizer | Learning scheduler |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Unet&EfficientNetB0 | 0.0799 | 40 | 250 | FocalLoss | Adam (lr=1e-3) | ReduceLROnPlateau(factor=0.5, patience=3) |
