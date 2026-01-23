import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import os 

train_file_path = 'drive/MyDrive/mitbih_train.csv'
test_file_path = 'drive/MyDrive/Datasets/mitbih_test.csv'
if not os.path.exists(train_file_path):
    print(f"Error: Training file not found at {train_file_path}. Please ensure it's in your Google Drive.")

if not os.path.exists(test_file_path):
    print(f"Error: Test file not found at {test_file_path}. Please ensure it's in your Google Drive.")
  

# Load data if files exist
if os.path.exists(train_file_path) and os.path.exists(test_file_path):
    train_df = pd.read_csv(train_file_path, header=None)
    test_df = pd.read_csv(test_file_path, header=None)

    #  Preprocessing
    # The last column is usually the class label 
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Reshape for LSTM: [samples, time_steps, features] -> [N, 187, 1]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # One-hot encode targets 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Build LSTM Model
    model = Sequential()

    # CNN Layers for Feature Extraction 
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax')) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #  Train
    print("Training ECG Model (This may take time)...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save
    # Ensure the directory exists before saving
    save_dir = 'models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(os.path.join(save_dir, 'ecg_lstm_model.h5'))
    print(f"ECG model saved to '{os.path.join(save_dir, 'ecg_lstm_model.h5')}'")
else:
    print("Model training and saving skipped due to missing data files.")