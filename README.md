# DTC Classifier
Dependency tree-based text classifier.

Requirements:
	Python:
		gensim 2.3.0
		nltk 3.2.4
		numpy 1.12.1
		scikit-learn 0.18.2
		tensorflow 1.0.0
	Java:
		json-simple-1.1.1
		stanford-corenlp-3.7.0
		stanford-corenlp-3.7.0-models

Files:
CNNClassifier.py - Convolutional neural network text classifier modified from from https://github.com/dennybritz/cnn-text-classification-tf.
DependencyTreeClassifier.py - Dependency tree-based text classifier.
DTCHelper/src/DTCHelper.java - NLP server for dependency trees and semgrex patterns. Uses Stanford CoreNLP.
DTCHelper/src/SemgrexPatternWrapper.java - Wrapper class for semgrex patterns.
extractPatterns.py - Pattern extraction algorithm.
RandomForestTextClassifier.py - Random forest text classifier. Uses SciKit-Learn's random forest classifier, wrapped with text-specific functionality.
TextClassifierUtils.py - Utility objects, including dataset loading functionality.

Instructions:
1. Start the Java NLP server (DTCHelper).
2. Load text datasets with TextDatasetFileParser.parse() in TextClassifierUtils.py. Dataset files can be .arff files with one string attribute, one nominal class attribute and an optional weight; or .csv files with format "text,class_label,weight" (weight is optional). This will return datasets in the form of a list of Instance objects.
3. Train the classifier (DependencyTreeClassifier in DependencyTreeClassifier.py) on training data with the classifier's train() method.
4. Evaluate the classifier's performance with the classifier's evaluate() method.

To change the "backup" classifier used by the DTC Classifier (used when no pattern matches a test sentence), pass either "rf" or "cnn" to the backup_classifier parameter of the DTC classifier's constructor to use a random forest or CNN classifier with default settings, or pass an object of type AbstractTextClassifier (from TextClassifierUtils.py) to the backup_classifier parameter.

To use the Word2Vec version of the CNN classifier, pass the path (string) to a text file containing one document per line to the unlabeled_data parameter of the CNN classifier's constructor.
