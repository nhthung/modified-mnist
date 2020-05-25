# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import os.path as op
from PIL import Image, ImageOps
from pathlib import Path
from keras.datasets import mnist

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_DIR / 'results'
SRC_DIR = PROJECT_DIR / 'src'
DATA_DIR = PROJECT_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

def main():
    """
    Generate and save MNIST images and labels to train
    "You Only Look Once" (YOLO) object detection system
    """
    logger = logging.getLogger(__name__)
    logger.info('preprocessing data')

    generate_mnist()

def generate_mnist():
    """
    Generate regular MNIST images and labels
    """
    logger = logging.getLogger(__name__)
    logger.info(('generating regular MNIST images and labels to train '
                 '"You Only Look Once" (YOLO) object detection system'))

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    write_config() # Write config files for YOLO training
    save_mnist_set(X_train, y_train, set_name='train')
    save_mnist_set(X_test, y_test, set_name='test')

def write_config():
    """
    Write .data and .names config files for YOLO training
    """
    logger = logging.getLogger(__name__)
    logger.info(('writing config files for YOLO training'))

    mnist_dir = PROCESSED_DIR / 'mnist'
    cfg_dir = SRC_DIR / 'cfg'
    config = ('classes = 10\n'
                f'train = {mnist_dir}/train.txt\n'
                f'valid = {mnist_dir}/test.txt\n'
                f'names = {mnist_dir}/voc-mnist.names\n'
                f'backup = {PROJECT_DIR}/models')

    # Create directories if don't exist already
    make_directory(cfg_dir)
    make_directory(mnist_dir)

    logger.info('writing voc-mnist.data')
    with open(cfg_dir / 'voc-mnist.data', 'w') as fout:
        fout.write(config)

    logger.info('writing voc-mnist.names')
    with open(mnist_dir / 'voc-mnist.names', 'w') as fout:
        fout.write('\n'.join(f'{i}' for i in range(10))) # Write 0-9

def save_mnist_set(X, y, set_name):
    """
    Generate and save MNIST images and labels to train
    "You Only Look Once" (YOLO) object detection system
    """
    logger = logging.getLogger(__name__)
    mnist_dir = PROCESSED_DIR / 'mnist'
    dataset_dir = mnist_dir / set_name

    make_directory(dataset_dir / 'img')
    make_directory(dataset_dir / 'labels')

    for i, x in enumerate(X):
        filename = f'{i:05d}'
        image, box = get_image_box_pair(x)
        label = get_label(digit_label=y[i], image=image, box=box)

        # Write image file
        logger.info(f'saving img/{filename}.jpg')
        image_path = dataset_dir / f'img/{filename}.jpg'
        image.save(image_path)

        # Write label file
        logger.info(f'saving labels/{filename}.txt')
        with open(dataset_dir / f'labels/{filename}.txt', 'w') as fout:
            fout.write(label)

        # Write list file (for YOLO training)
        with open(mnist_dir / f'{set_name}.txt', 'a') as fout:
            fout.write(f'{image_path}\n')

def get_image_box_pair(x):
    """
    MNIST digit image to train "You Only Look Once" (YOLO) object detection system

    Outputs:
        image: PIL.Image - Handwritten digit in a square bounding box
                           on a rectangular background

        box: dictionary - Dimensions and position of the digit's bounding box
                          on the rectangular background
    """
    blue = (
        0x00, # Red
        0x00, # Green
        0xff, # Blue
    )
    background = Image.new('RGB', size=(500, 375), color=blue)
    image = Image.fromarray(x.astype('uint8'))

    box_width = image.width * 10
    box_height = image.height * 10
    box_left = (background.width - box_width) // 2
    box_top = (background.height - box_height) // 2
    box = { # Dimensions & position of box on rectangular background
        'left': box_left,
        'right': box_left + box_width,
        'top': box_top,
        'bottom': box_top + box_height,
    }

    image = image.resize(size=(box_width, box_height)) # Enlarge image
    image = ImageOps.invert(image) # Invert colors
    background.paste(image, box=(box_left, box_top)) # Paste digit onto background
    image = background

    return image, box

def get_label(digit_label, image, box):
    """
    MNIST digit label to train "You Only Look Once" (YOLO) object detection system
    """
    x = (box['left'] + box['right']) / 2
    y = (box['top'] + box['bottom']) / 2
    w = box['right'] - box['left']
    h = box['bottom'] - box['top']
    x /= image.width
    w /= image.width
    y /= image.height
    h /= image.height
    return f'{digit_label:d}, {x:f}, {y:f}, {w:f}, {h:f}'

def make_directory(path):
    """
    Create directory if it doesn't already exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def get_dataset(path):
    return pd.read_pickle(path)

def get_csv(path):
    return pd.read_csv(path).to_numpy()[:, 1]

def load_train():
    train_images = pd.read_pickle(op.join(DATA_DIR, 'train_images.pkl'))
    train_labels = pd.read_csv(op.join(DATA_DIR, 'train_labels.csv')).to_numpy()[:, 1]
    return train_images, train_labels

def load_test():
    return pd.read_pickle(op.join(DATA_DIR, 'test_images.pkl'))

def predictions_to_csv(pred, filename):
    if not op.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    res = pd.Series(pred, name='Category')
    submission = pd.concat([pd.Series(range(0, res.shape[0]), name='Id'), res], axis=1)
    submission.to_csv(op.join(RESULT_DIR, filename), index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
