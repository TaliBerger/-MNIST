%load_ext tensorboard
import datetime, os
import tensorflow as tf
import tensorflow_datasets as tfds

# טעינת הנתונים
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

num_of_classes = ds_info.features['label'].num_classes
image_shape = ds_info.features['image'].shape
train_size = ds_info.splits['train'].num_examples
test_size = ds_info.splits['test'].num_examples

print(num_of_classes)
print(image_shape)
print(train_size)
print(test_size)

#פונקציית נירמול
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# נירמול הנתונים
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test=ds_test.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)
ds_test=ds_test.batch(128)
ds_test=ds_test.cache()
ds_test=ds_test.prefetch(tf.data.AUTOTUNE)


#ניסוי 11
#בניית המודל
layers = [
  #  tf.keras.Sequential([
   tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(93, activation='relu'),# שכבה ראשונה
      tf.keras.layers.Dropout(0.3),# שכבת Dropout
    tf.keras.layers.Dense(16, activation='relu'), # שכבה שנייה
    tf.keras.layers.Dense(92, activation= 'relu'), # שכבה שלישית
    tf.keras.layers.Dense(14, activation= 'relu'), # שכבה רביעית
    tf.keras.layers.Dense(10, activation='softmax'), # הגדרה של 10 קטגוריות
]

# קימפול המודל
model = tf.keras.models.Sequential(layers)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
model.summary()


# אימון המודל
model.fit(ds_train, epochs=10,validation_data=ds_test)

# בדיקת המודל
model.evaluate(ds_test)

"""
תוצאה הכי טובה,Epoch=9

1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - loss: 0.1294 - sparse_categorical_accuracy: 0.9636 - val_loss: 0.1034 - val_sparse_categorical_accuracy: 0.9739
"""
