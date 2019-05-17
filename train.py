import torch
from data import get_train
import matplotlib.pyplot as plt

plt.ion()
use_gpu = torch.cuda.is_available()

def main():
    train_images, train_labels = get_train()
    if use_gpu:
        print('Using CUDA')

if __name__ == '__main__':
    main()
