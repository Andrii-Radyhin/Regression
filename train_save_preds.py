import pandas as pd
import matplotlib.pyplot as plt
from keras.metrics import RootMeanSquaredError
import keras
from tensorflow.keras import layers
import tensorflow as tf

df = pd.read_csv('internship_train.csv')
targets = df['target']
df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='target'),
    targets,
    test_size=0.001,
    shuffle=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='regression.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
]




normalizer = layers.Normalization(input_shape=[53,], axis=None)

def build_and_compile_model(norm):
    model = keras.Sequential([
      norm,
      layers.Dense(128, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(1)
  ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001), metrics = ['RootMeanSquaredError'] )
    return model

model = build_and_compile_model(normalizer)
#model.summary()

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

test = pd.read_csv('internship_hidden_test.csv')
preds = model.predict(test)
preds_df = pd.DataFrame(preds, columns=['preds'])
preds_df.to_csv('preds_df.csv')