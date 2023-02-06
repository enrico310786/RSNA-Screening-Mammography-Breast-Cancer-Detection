import pandas as pd

path_val_csv = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/final_train_val_split_augmented/val.csv"
val_df = pd.read_csv(path_val_csv).reset_index(drop=True)
print("val_df info")
print(val_df.info())
val_df.to_csv(path_val_csv, index=False)
