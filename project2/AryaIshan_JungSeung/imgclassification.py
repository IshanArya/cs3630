#!/usr/bin/env python
import numpy as np
import os.path
import pickle
import re
from sklearn import svm, metrics
from skimage import io, feature, filters, exposure, color
from skimage import transform

# io.use_plugin('pil')


class ImageClassifier:
    def __init__(self, modelPath=None):
        self.classifier = None
        if modelPath != None:
            self.load_classifier(modelPath)

    def load_classifier(self, filePath):
        file = open(filePath, "rb")
        serialModel = file.read()
        self.classifier = pickle.loads(serialModel)

    def save_classifier(self, filePath):
        file = open(filePath, "wb")
        serialModel = pickle.dumps(self.classifier)
        file.write(serialModel)

    def extract_image_features(self, data):
        feature_data = []
        for image in data:
            image = color.rgb2gray(image)
            image = transform.rescale(image, 1.0 / 4.0, multichannel=False)
            feature_data.append(feature.hog(image))

        return(feature_data)

    def train_classifier(self, train_data, train_labels):
        # train model and save the trained model to self.classifier
        self.classifier = svm.SVC(kernel='poly', gamma='scale', degree=2)
        self.classifier.fit(train_data, train_labels)

    def predict_labels(self, data):
        predicted_labels = self.classifier.predict(data)
        return predicted_labels

    def imread_convert(self, f):
        return io.imread(f).astype(np.uint8)

    def load_data_from_folder(self, dir):
        # read all images into an image collection
        ic = io.ImageCollection(dir+"*.bmp", load_func=self.imread_convert)

        # create one large array of image data
        data = io.concatenate_images(ic)

        # extract labels from image names
        labels = np.array(ic.files)
        for i, f in enumerate(labels):
            m = re.search("_", f)
            labels[i] = f[len(dir):m.start()]

        return(data, labels)


def main():

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder("./train/")
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    img_clf.save_classifier("model.txt")

    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(
        train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(
        train_labels, predicted_labels, average='micro'))

    # test model
    predicted_labels = img_clf.predict_labels(test_data)
    print("\nTest results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(
        test_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(test_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(
        test_labels, predicted_labels, average='micro'))


if __name__ == "__main__":
    main()
