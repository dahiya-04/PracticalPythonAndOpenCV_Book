from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from p.SimpleDatasetLoader import SimpleDatasetLoader
from p.preprocessor import SimplePreprocessor
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help ="path to input dataset")
ap.add_argument("-k","--neighbors",type = int,default=1,help ="# of nei. neighbor")
ap.add_argument("-j","--jobs",type =int,default=1,help="pert. of jobs for K-NN distance")

args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 *1000.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state=42)

# train and evaluate a K-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

