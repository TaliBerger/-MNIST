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
    with_info=True,
)

#פונקציית נירמול
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# נירמול הנתונים
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)

#ניסוי 1
#בניית המודל
model_1= tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(93, activation='relu'), # שכבה ראשונה
    tf.keras.layers.Dense(16, activation='relu'), # שכבה שנייה
    tf.keras.layers.Dense(92, activation= 'relu'), # שכבה שלישית
    tf.keras.layers.Dense(14, activation= 'relu'), # שכבה רביעית
    tf.keras.layers.Dense(10, activation='softmax'), # הגדרה של 10 קטגוריות
])

# קימפול המודל
model_1.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# אימון המודל
model_1.fit(ds_train, epochs=5)

# בדיקת המודל
model_1.evaluate(ds_test)

#תוצאות פלט ניסוי מספר 1
"""
Epoch 1/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 17s 8ms/step - loss: 0.6841 - sparse_categorical_accuracy: 0.7875
Epoch 2/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - loss: 0.1537 - sparse_categorical_accuracy: 0.9555
Epoch 3/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - loss: 0.1066 - sparse_categorical_accuracy: 0.9689
Epoch 4/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - loss: 0.0847 - sparse_categorical_accuracy: 0.9769
Epoch 5/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - loss: 0.0729 - sparse_categorical_accuracy: 0.9801
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.1191 - sparse_categorical_accuracy: 0.9687
[0.11359720677137375, 0.9692000150680542]
"""
