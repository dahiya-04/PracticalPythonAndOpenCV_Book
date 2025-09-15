# import necessary packages
from lenet import LeNet
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# grab the dataset
print("[INFO] accessing MNIST...")
dataset = fetch_openml("mnist_784", version=1)
data = dataset.data

# channels first
if K.image_data_format() == "channels_first":
    data = data.values.reshape(data.shape[0], 1, 28, 28)

# channels last
else:
    data = data.values.reshape(data.shape[0], 28, 28, 1)

# splitting the data and scaling
(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
    dataset.target.astype("int"), test_size=0.25, random_state=42)

# one-hot encoding of labels
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# compile the model
print("[INFO] compiling the model...")
opt = SGD(learning_rate=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=128, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating the network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
