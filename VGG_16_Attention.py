import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class nn_model:
  early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
  class celeb:
    model_checkpoint = ModelCheckpoint("./dl_models/smr_celeb_df_DLIB_CNN_SGD.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1)
  class ffpp:
    model_checkpoint = ModelCheckpoint("./dl_models/smr_ffpp_DLIB_CNN_SGD.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

  def VGG_16_with_attention(self, input_shape=(176, 208, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # Block 2
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # Block 3
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Block 4
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Block 5
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # Flatten spatial dimensions
    batch_size, height, width, channels = x.shape
    x = layers.Reshape((height * width, channels))(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=512, dropout=0.1)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    # Classification head
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
     )
    model.summary()
    return model

