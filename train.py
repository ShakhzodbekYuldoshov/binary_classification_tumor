# Import required libraries
from get_model import get_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.metrics import Precision, Recall


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('./dataset/train',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    shuffle=True)
test_generator = test_datagen.flow_from_directory('./dataset/test',
                                                  target_size=(224, 224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle=True)

model = get_model(tf)

checkpoint_path = "training_3/cp.ckpt"
model.load_weights(checkpoint_path)

print(model.summary())

for layer in model.layers[:175]:
    layer.trainable = False

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

# save model
checkpoint_path = "training_4/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# draw model metrics
history = model.fit_generator(
    generator=train_generator, 
    steps_per_epoch=train_generator.n//train_generator.batch_size, 
    epochs=20,
    callbacks=[cp_callback, es_callback])

print(history.history.keys())
acc = history.history['acc']
loss = history.history['loss']
plt.figure()
plt.plot(acc, label='Training Accuracy')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.figure()
plt.plot(loss, label='Training Loss')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()

# evaluate the model
Testresults = model.evaluate(test_generator)
print("test loss, test acc:", Testresults)

predictions = model.predict(test_generator, batch_size=None, verbose=0, steps=None, callbacks=None)
print(predictions)

classes = np.argmax(predictions, axis = 1)
print(classes)
print(test_generator.labels)

x=test_generator[5]
predictions = model.predict(x) 
print('First prediction:', predictions[0])
category = np.argmax(predictions[0], axis = 0)
print("the image class is:", category)

