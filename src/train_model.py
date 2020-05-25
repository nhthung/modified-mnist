from keras.utils import to_categorical, plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Adadelta
from sklearn.model_selection import train_test_split
from keras import backend as K
from matplotlib import pyplot as plt

from data import load_train
from models.cnn_vgg import VGG

print(K.tensorflow_backend._get_available_gpus())

# Data properties
num_classes = 10
img_x, img_y = 64, 64

# Load training data
train_images, train_labels = load_train()
x_train, x_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Reshape and normalize images
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_x, img_y, 1)
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255.
x_valid /= 255.

# One-hot encode labels
y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)

print(f'Train images dim: {x_train.shape}')
print(f'Train labels dim:{y_train.shape}')
print(f'Validation images dim: {x_valid.shape}')
print(f'Validation labels dim:{y_valid.shape}')

# Best: Ghetto vgg10 (half layer dimensions), Adam optimizer

# optimizer = Adadelta()
optimizer = Adam()
# optimizer = Nadam()

# Train params
batch_size = 64
num_steps = 'auto'
# num_steps = 1000
epochs = 100
model_name = 'VGG9'

# Callbacks
annealer = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5)
early_stop = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
checkpoint = ModelCheckpoint(
    filepath=f'./project3/trained_models/checkpoints/{model_name}_steps={num_steps}_batch={batch_size}.h5',
    verbose=1,
    monitor='val_acc',
    save_best_only=True)

model = VGG(input_shape=(img_x, img_y, 1), num_classes=num_classes, optimizer=optimizer)
plot_model(model.model, to_file=f'./project3/figures/{model_name}_arch.png', show_shapes=True)

history = model.train(
    x_train, y_train,
    x_valid, y_valid,
    batch_size=batch_size,
    epochs=epochs,
    datagen=True,
    num_steps=num_steps,
    callbacks=[annealer, early_stop, checkpoint]
    )

score = model.evaluate(x_valid, y_valid)

print(f'Validation Loss: {score[0]}')
print(f'Validation Accuracy: {score[1]}')

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']

if score[1] >= 0.97:
    model.save(f'./project3/trained_models/{model_name}_{round(score[1]*100, 2)}%.h5')
else:
    print('Model did not exceed baseline validation accuracy, not saving.')

epcs = range(1, len(acc) + 1)
plt.plot(epcs, acc, 'bo', label='Training acc')
plt.plot(epcs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'./project3/figures/{model_name}_{round(score[1]*100, 2)}%.png')
plt.show()
