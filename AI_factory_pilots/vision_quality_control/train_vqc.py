
"""Train a small CNN (Keras) on the synthetic defect dataset."""
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_generator import generate_dataset

def build_model(input_shape=(64,64,1)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_dir = "vqc_data"
    if not os.path.exists(data_dir):
        generate_dataset(out_dir=data_dir, n_pos=1200, n_neg=2800, size=(64,64))
    batch_size = 32
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                       rotation_range=0, width_shift_range=0.05, height_shift_range=0.05,
                                       horizontal_flip=True)
    train_gen = train_datagen.flow_from_directory(data_dir, target_size=(64,64), color_mode='grayscale',
                                                  batch_size=batch_size, class_mode='binary', subset='training', shuffle=True)
    val_gen = train_datagen.flow_from_directory(data_dir, target_size=(64,64), color_mode='grayscale',
                                                batch_size=batch_size, class_mode='binary', subset='validation', shuffle=True)
    model = build_model((64,64,1))
    os.makedirs("artifacts", exist_ok=True)
    checkpoint = ModelCheckpoint("artifacts/vqc_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    early = EarlyStopping(monitor='val_accuracy', patience=6, mode='max', restore_best_weights=True)
    history = model.fit(train_gen, epochs=25, validation_data=val_gen, callbacks=[checkpoint, early])
    # Save final model
    model.save("artifacts/vqc_model.h5")
    print("Training complete. Model saved to artifacts/vqc_model.h5")

if __name__ == '__main__':
    main()
