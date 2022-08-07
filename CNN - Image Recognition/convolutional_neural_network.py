import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# ********************************************************************************************
# DATA PREPROCESSING
# training set: rescale and augmentations needed to avoid overfiting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# test set: no agumentations needed, same rescaling still needed
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# ********************************************************************************************
# CNN
cnn = tf.keras.models.Sequential()

# step 1: convolution operation
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,   # 3x3 feature maps
    activation='relu',
    input_shape=[64, 64, 3]  # only needed on first layer
))

# step 2: max-pooling
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2),
    strides=2
))

# second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

# step 3: flattening
cnn.add(tf.keras.layers.Flatten())

# step 4: full connection and output layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# training
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=test_set, epochs=25)


# ********************************************************************************************
# PREDICTION
print('Please enter desired prediction image source or enter \'q\' to exit the program')
pred_opt = input(
    ' Enter a number between 1-4 to use one of the 4 preset predictions or enter an absolute path to the picture you would like to use:')

while (pred_opt != 'q'):
    test_image = ''
    if pred_opt.__len__() == 1:
        img_filepath = 'dataset/single_prediction/cat_or_dog_' + pred_opt + '.jpg'
        test_image = tf.keras.preprocessing.image.load_img(
            img_filepath, target_size=(64, 64))
        print(img_filepath)
    else:
        test_image = tf.keras.preprocessing.image.load_img(
            pred_opt, target_size=(64, 64))

    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = cnn.predict(test_image/255.0)
    training_set.class_indices
    if result[0][0] > 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'

    print('The picture likely contains a: ' + prediction)
    pred_opt = input(
        'Please enter 1-4 for another preset test file or an absolute file path to a test image, or \'q\' to exit the program: ')
