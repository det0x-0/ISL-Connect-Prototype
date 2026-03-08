import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1. Configuration
DATA_PATH = r"C:\Users\User\Desktop\aslinterpretationusingdeeplearning\MP_Data"
actions = np.array(['hello', 'my', 'name', 'thanks', 'A', 'S', 'H', 'I', 'Q'])
sequence_length = 30 # Number of frames per video remains 30

label_map = {label:num for num, label in enumerate(actions)}

# 2. Load and Validate Data
sequences, labels = [], []
print("Loading data...")
print(f"Expected shape per frame: 1662")

# Track loading issues
loading_errors = []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"❌ WARNING: Folder not found for action '{action}'")
        continue
    
    # --- CHANGE: Get all numeric subfolders in the action directory ---
    # This finds every sequence folder (0, 1, 2... up to 119, etc.)
    sequence_folders = [d for d in os.listdir(action_path) 
                        if os.path.isdir(os.path.join(action_path, d)) and d.isdigit()]
    
    # Sort them numerically so they load in order (0, 1, 2...)
    sequence_folders.sort(key=int)
    
    loaded_count = 0
    for sequence in sequence_folders:
        window = []
        try:
            for frame_num in range(sequence_length):
                file_path = os.path.join(action_path, sequence, f"{frame_num}.npy")
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Missing: {file_path}")
                
                res = np.load(file_path)
                
                # Validate shape
                if res.shape[0] != 1662:
                    raise ValueError(f"Wrong shape {res.shape} in {file_path}")
                
                window.append(res)
            
            sequences.append(window)
            labels.append(label_map[action])
            loaded_count += 1
            
        except Exception as e:
            loading_errors.append(f"{action}/{sequence}: {str(e)}")
    
    print(f"✅ Loaded {loaded_count} sequences for '{action}'")

if loading_errors:
    print(f"\n⚠️ Found {len(loading_errors)} loading errors:")
    for err in loading_errors[:5]:  # Show first 5
        print(f"  - {err}")
    if len(loading_errors) > 5:
        print(f"  ... and {len(loading_errors) - 5} more")

# Convert to arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"\n📊 Data Summary:")
print(f"Total sequences: {len(X)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {actions.shape[0]}")

# Check for NaN or Inf
if np.isnan(X).any():
    print("⚠️ WARNING: NaN values detected in data!")
if np.isinf(X).any():
    print("⚠️ WARNING: Inf values detected in data!")

# 3. Split Data (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\n📚 Split sizes:")
print(f"Training: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Testing: {len(X_test)}")

# 4. Build Model with Dropout for regularization
print("\n🏗️ Building model...")
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# 5. Callbacks
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=0.00001,
    verbose=1
)

# 6. Train
print("\n🚀 Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    callbacks=[tb_callback, early_stop, reduce_lr],
    batch_size=32,
    verbose=1
)

# 7. Evaluate on Test Set
print("\n📈 Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# 8. Per-class accuracy
predictions = model.predict(X_test, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("\n📊 Per-Class Accuracy:")
for i, action in enumerate(actions):
    mask = true_classes == i
    if mask.sum() > 0:
        acc = (pred_classes[mask] == i).sum() / mask.sum()
        print(f"  {action:10s}: {acc*100:5.1f}% ({mask.sum()} samples)")

# 9. Save Model
model.save('asl_model1.h5')
weights = np.array(model.get_weights(), dtype=object)
np.save('asl_weights1.npy', weights)
print("\n✅ Model saved as asl_model.h5 and asl_weights.npy")

# 10. Plot Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Train Acc')
plt.plot(history.history['val_categorical_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("📊 Training plots saved as training_history.png")

# 11. Diagnostic Summary
print("\n" + "="*50)
print("DIAGNOSTIC SUMMARY")
print("="*50)
print(f"✅ Final Training Accuracy: {history.history['categorical_accuracy'][-1]*100:.2f}%")
print(f"✅ Final Validation Accuracy: {history.history['val_categorical_accuracy'][-1]*100:.2f}%")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

gap = history.history['categorical_accuracy'][-1] - history.history['val_categorical_accuracy'][-1]
if gap > 0.15:
    print(f"⚠️ Large train-val gap ({gap*100:.1f}%) - Model may be overfitting")
elif test_acc < 0.7:
    print(f"⚠️ Low test accuracy - Model needs more data or better features")
else:
    print(f"✅ Model looks healthy!")

print("\nNext steps:")
print("1. Check training_history.png to see if model is learning")
print("2. If test accuracy is low (<70%), you may need:")
print("   - More training data")
print("   - Better quality recordings")
print("   - More distinct signs")
print("3. Run the live detection script to test in real-time")