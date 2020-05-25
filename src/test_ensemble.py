import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Input
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from models.cnn_ensemble import Ensemble
from data import load_train, load_test, predictions_to_csv

# model_names = ['cnn_ens1_96.02%', 'cnn_ens2_96.2%', 'cnn_ens3_96.45%']
# model_names = ['cnn_ens1_96.02%', 'cnn_ens2_96.2%', 'cnn_ens3_96.45%', 'cnn_ens4_96.76%', 'cnn_ens5_96.72%',
#                'cnn_ens6_96.51%']
model_names = ['cnn_ens1_96.02%', 'cnn_ens2_96.2%', 'cnn_ens3_96.45%', 'cnn_ens4_96.76%', 'cnn_ens5_96.72%',
'cnn_ens6_96.51%', 'cnn_ens7_96.15%', 'cnn_ens8_96.29%', 'cnn_ens9_96.34%', 'cnn_ens10_96.26%']

models = []
for i, name in enumerate(model_names):
    model = load_model(f'./project3/trained_models/{name}.h5')
    model.name = f'ensemble_{i+1}'
    models.append(model)

model_input = Input(shape=(64, 64, 1))
model = Ensemble(models, model_input)
model_name = 'ensemble'

print(model.summary())
plot_model(model.model, to_file=f'./project3/figures/{model_name}{len(model_names)}_arch.png', show_shapes=True)

num_classes = 10
img_x, img_y = 64, 64

# Load data
train_images, train_labels = load_train()
x_test = load_test()

_, x_valid, _, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Reshape and normalize
x_valid = x_valid.reshape(x_valid.shape[0], img_x, img_y, 1)
x_valid = x_valid.astype('float32')
x_valid /= 255.
print('Validation dim: ', x_valid.shape)

x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
x_test = x_test.astype('float32')
x_test /= 255.
print('Test dim: ', x_test.shape)

# Compute validation accuracy
y_val_pred = model.predict(x_valid)
y_val_pred = np.argmax(y_val_pred, axis=1)
print(y_val_pred.shape)

num_correct = 0
for y, y_pred in zip(y_valid, y_val_pred):
    if y == y_pred:
        num_correct += 1
print('Validation Accuracy: ', num_correct/y_valid.shape[0])

# Generate test labels
y_test = model.predict(x_test)
y_test = np.argmax(y_test, axis=1)
print(y_test.shape)
predictions_to_csv(y_test, f'{model_name}_pred.csv')

plt.show()
