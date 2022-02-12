from get_model import get_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory('./dataset/test',
                                                  target_size=(224, 224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle=True)

model = get_model(tf)

checkpoint_path = "training_4/cp.ckpt"
model.load_weights(checkpoint_path)

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_results = model.evaluate(test_generator)
print("test loss, test acc:", test_results)

predictions = model.predict(test_generator, batch_size=None, verbose=0, steps=None, callbacks=None)

predictions = np.argmax(predictions, axis = 1)

print(predictions)

print(test_generator.labels)

print('accuracy score:   ', accuracy_score(test_generator.labels, predictions))
# sns.heatmap(confusion_matrix(test_generator.labels, predictions),  annot=True)
# plt.show()

print(test_generator.labels, test_generator)

# print('hello world', dir(tf.math.confusion_matrix(test_generator.labels, predictions)))
print('hello world',confusion_matrix(test_generator.labels, predictions))
print(precision_recall_fscore_support(test_generator.labels, predictions, average='macro'))
