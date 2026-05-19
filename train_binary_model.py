import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
IMG_SIZE = (64, 64) # Smaller size works well for binary CNN
BATCH_SIZE = 32
DATASET_DIR = 'datasets'

# --- Data Augmentation ---
# We still use this to make the 150 images feel like 1000
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

print("Loading Training Data...")
train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # <--- BINARY MODE
    subset='training'
)

print("Loading Validation Data...")
val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# --- Custom Binary CNN Architecture ---
model = Sequential([
    Input(shape=(64, 64, 3)),
    
    # Layer 1
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Helps prevent memorizing
    
    # Output Layer: 1 Neuron, Sigmoid Activation (0 to 1)
    Dense(1, activation='sigmoid') 
])

# --- Compile ---
model.compile(optimizer='adam', 
              loss='binary_crossentropy', # Essential for binary
              metrics=['accuracy'])

# --- Train ---
print("🚀 Starting Binary Training...")
model.fit(train_gen, validation_data=val_gen, epochs=15)

# --- Save ---
model.save("activity_binary_model.h5")
print("✅ Model saved as 'activity_binary_model.h5'")

# Print class mapping so we know which is 0 and which is 1
print(f"✅ Class Mapping: {train_gen.class_indices}")