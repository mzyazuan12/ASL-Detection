import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD

# dataset path
input_dir = './processed_data'
batch_size = 32
image_size = (224, 224)

# load dataset and retrieve class names
train_data = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_data.class_names  

# normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# optimize data loading
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_size + (3,))

# unfreeze some layers for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Sequential([
    base_model,  #VGG16 model
    GlobalAveragePooling2D(),
    Dropout(0.3),  
    Dense(128, activation='relu'),
    Dropout(0.3),  
    Dense(len(class_names), activation='softmax')  
])

model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[lr_scheduler]
)


model.save('./models/asl_model.h5')
