import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from utils.visualize import plot_training_history, evaluate_model

# ==== CẤU HÌNH ====
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(4)

# ==== THAM SỐ ====
IMG_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 64, 7
DROPOUT, AUTOTUNE = 0.1, tf.data.AUTOTUNE
LR1, LR2 = 2e-3, 3e-5
EPOCHS1, EPOCHS2 = 15, 45

# ==== AUGMENTATION & NORMALIZATION ====
data_aug = tf.keras.Sequential([
    layers.RandomZoom(0.1),
])
normalize = layers.Rescaling(1./255)

# ==== LOAD DATASET ====
def prepare_ds(path, training=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, label_mode='categorical',
        shuffle=training
    )
    if training:
        ds = ds.map(lambda x, y: (data_aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE).prefetch(1)

train_ds = prepare_ds("dataset/train", training=True).shuffle(100)
val_ds = prepare_ds("dataset/test", training=False)

# ==== MODEL ====
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
for units in [256, 128]:
    x = layers.Dense(units, activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

# ==== GIAI ĐOẠN 1 (HUẤN LUYỆN PHẦN MỀM) ====
model.compile(optimizer=AdamW(learning_rate=LR1, weight_decay=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1)

# ==== GIAI ĐOẠN 2 (FINE-TUNING) ====
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=AdamW(learning_rate=LR2, weight_decay=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Callback lưu model tốt nhất
checkpoint_callback = ModelCheckpoint(
    "models/best_model.keras",
    monitor='val_loss',
    save_best_only=True
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2,
                    callbacks=[checkpoint_callback])

# ==== LƯU & ĐÁNH GIÁ ====
# Lưu model sau cùng
model.save("models/final_model.keras")

# Vẽ biểu đồ huấn luyện và đánh giá
plot_training_history(history)
evaluate_model(model, val_ds)
