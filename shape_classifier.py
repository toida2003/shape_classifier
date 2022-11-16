import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow
import numpy as np
import glob
from sklearn.model_selection import train_test_split
tf = tensorflow.compat.v1
tf.disable_eager_execution()
kr = tf.keras

folders = ["circle", "triangle", "square"]
outfolder_end = "_shaping"

dataset_x = []
dataset_y = []

for index_folder, folder in enumerate(folders):
    files = glob.glob("img/" + folder + outfolder_end + "/*.bmp", recursive = True)
    for file in files:
        dataset_x.append(
            kr.preprocessing.image.img_to_array(
                kr.preprocessing.image.load_img(
                    file, 
                    target_size=(20, 20), 
                    color_mode="grayscale"
                )
            )
        )
        dataset_y.append(index_folder)

dataset_x = np.asarray(dataset_x)
dataset_y = np.asarray(dataset_y)
print("X shape:", dataset_x.shape)
print("y shape:", dataset_y.shape)

dataset_x = dataset_x / 255
dataset_y = kr.utils.to_categorical(dataset_y, len(folders))

x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, random_state = 104)


model = kr.models.Sequential()

for i in range(1):
    model.add(
        kr.layers.Conv2D(
            filters=16,
            input_shape=(20, 20, 1),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu"
        )
    )

    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(kr.layers.Dropout(0.25))

model.add(kr.layers.Flatten())

model.add(kr.layers.Dense(units=100, activation="relu"))
model.add(kr.layers.Dropout(0.5))
model.add(kr.layers.Dense(units=len(folders), activation="softmax"))

print(model.output_shape)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

log_dir = "logs/fit/"
tsb = kr.callbacks.TensorBoard(log_dir=log_dir)

history_model = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=15,
    validation_split=0.2,
    callbacks=[tsb]
)

model.save("model/shape_classifier.h5")