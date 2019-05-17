# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parents[0]
data_dir = project_dir / 'data'

def main():
    pass

def get_train():
    train_images = pd.read_pickle(data_dir / 'train_images.pkl')
    train_labels = pd.read_csv(data_dir / 'train_labels.csv')
    return train_images, train_labels

if __name__ == '__main__':
    main()
