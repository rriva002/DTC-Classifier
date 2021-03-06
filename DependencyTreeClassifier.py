from CNNClassifier import CNNClassifier
from collections import Counter
from extractPatterns import PatternExtractor
from json import dumps, loads
from math import log
from RandomForestTextClassifier import RandomForestTextClassifier
from sklearn.feature_extraction.text import CountVectorizer
from socket import AF_INET, SOCK_STREAM, socket
from TextClassifierUtils import AbstractTextClassifier, Instance


class NlpService(object):
    """
    Handler for communication with the NLP server (DTCHelper).

    Constructor arguments:

    host (default "localhost") - Host for the NLP server.

    port (default 9000) - Port for the NLP server.
    """
    def __init__(self, host="localhost", port=9000):
        self.__sock = socket(AF_INET, SOCK_STREAM)

        self.__sock.connect((host, port))

    def __receive(self):
        data = ""

        while len(data) == 0 or not data[len(data) - 1] == "\n":
            data += self.__sock.recv(1).decode("latin-1")

        return loads(data[0:len(data) - 1].replace("__NEWLINE__", "\n"))

    def __send(self, command, data):
        data["command"] = command

        self.__sock.send((dumps(data) + "\n").encode())

    def add_pattern(self, pattern, class_value):
        self.__send("add_pattern", {"pattern": pattern, "class": class_value})

    def classify(self, text, class_value=None):
        self.__send("classify", {"text": text, "class": class_value})
        return self.__receive()

    def end(self):
        self.__send("end", {})

    def has_pattern(self, pattern, class_value):
        self.__send("has_pattern", {"pattern": pattern, "class": class_value})
        return self.__receive()

    def parse(self, text):
        self.__send("parse", {"text": text})
        return self.__receive()

    def set_mode(self, mode):
        self.__send("set_mode", {"mode": mode})

    def test(self, text, class_value):
        self.__send("test", {"text": text, "class": class_value})


class DefaultRuleClassifier(AbstractTextClassifier):
    """
    Simple classifier that classifies text as the most common class label in
    the training data.
    """
    def __init__(self):
        self.__default_distribution = {}

    def classify(self, text):
        return self.__default_distribution

    def train(self, data):
        classes = []

        for instance in data:
            classes.append(instance.class_value)

        most_common = Counter(classes).most_common(1)[0][0]

        for class_label in set(classes):
            self.__default_distribution[class_label] = 1 if \
                class_label == most_common else 0


class DependencyTreeClassifier(AbstractTextClassifier):
    """
    Dependency tree-based text classifier. Constructs semgrex patterns
    extracted from dependency trees generated from training data.

    Constructor arguments:

    backup_classifier (default None) - Classifier to use if no pattern matches
    a sentence. If None, DefaultRuleClassifier is used. If "cnn", a CNN
    classifier is used. If "rf," a random forest classifier is used. If a
    text classifier (based on AbstractTextClassifier in TextClassifierUtils.py)
    is passed as an argument, that classifier will be used.

    nlp_host (default "localhost") - Host for the NLP server (DTCHelper).

    num_words (default 10) - Number of words to make available to pattern
    extraction. Words are selected by their information gain. May be a
    dictionary to specify the number of words for each non-neutral class label.

    max_words (default 4) - The maximum number of words that may be in a single
    semgrex pattern.
    """
    def __init__(self, backup_classifier=None, nlp_host="localhost",
                 num_words=10, max_words=4):
        self.__nlp = NlpService(host=nlp_host)
        self.__num_words = num_words
        self.__max_words = max(1, max_words)

        if backup_classifier is None:
            self.__backup_classifier = DefaultRuleClassifier()
        elif backup_classifier == "cnn":
            self.__backup_classifier = CNNClassifier()
        elif backup_classifier == "rf":
            self.__backup_classifier = RandomForestTextClassifier()
        else:
            self.__backup_classifier = backup_classifier

    def __entropy(self, vectors, index):
        value_frequencies = {}
        entropy = 0

        for vector in vectors:
            if vector[index] not in value_frequencies:
                value_frequencies[vector[index]] = 0

            value_frequencies[vector[index]] += 1

        for frequency in value_frequencies.values():
            ratio = frequency / len(vectors)
            entropy -= ratio * log(ratio, 2)

        return entropy

    def __top_information_gain_words(self, data, top_k):
        vectorizer = CountVectorizer(stop_words="english")
        text = []
        ig_values = []
        top_words = []
        word_frequencies = {}

        for instance in data:
            text.append(instance.text)

        vector_array = vectorizer.fit_transform(text).toarray()
        vectors = []
        words = vectorizer.get_feature_names()
        class_index = len(words)

        for i in range(0, len(data)):
            vector = []

            for value in vector_array[i]:
                vector.append(value)

            vector.append(data[i].class_value)
            vectors.append(vector)

        entropy = self.__entropy(vectors, class_index)

        for i in range(0, len(words)):
            word_frequencies[i] = {}

        for vector in vectors:
            for i in range(0, len(vector) - 1):
                if vector[i] not in word_frequencies[i]:
                    word_frequencies[i][vector[i]] = 0

                word_frequencies[i][vector[i]] += 1

        for index, value_frequencies in word_frequencies.items():
            subset_entropy = 0

            for value, frequency in value_frequencies.items():
                value_probability = frequency / sum(value_frequencies.values())
                sub = [vector for vector in vectors if vector[index] == value]
                subset_entropy += value_probability * \
                    self.__entropy(sub, class_index)

            ig_values.append((words[index], entropy - subset_entropy))

        ig_values.sort(key=lambda word: word[1], reverse=True)

        limit = min(top_k, len(ig_values)) if top_k > 0 else len(ig_values)

        for i in range(0, limit):
            top_words.append(ig_values[i][0])

        return top_words

    def classify(self, instance):
        distribution = self.__nlp.classify(instance.text, instance.class_value)
        distribution = self._normalize_distribution(distribution)

        return distribution if len(distribution) > 0 else \
            self.__backup_classifier.classify(instance)

    # Disconnect from the NLP server.
    def disconnect(self):
        if self.__nlp is not None:
            self.__nlp.end()

    def train(self, data):
        classes = []

        # Determine the most common class label.
        for instance in data:
            classes.append(instance.class_value)

        counter = Counter(classes)
        most_common = counter.most_common(1)[0][0]
        classes = set(classes)

        classes.remove(most_common)
        self.__nlp.set_mode("train")
        self.__backup_classifier.train(data)

        pattern_extractor = PatternExtractor(self.__nlp, self.__max_words)

        for class_value in classes:
            binary_data = []
            trees = []

            # Convert training sentences to dependency trees and determine the
            # top information gain words for the current class value.
            for instance in data:
                text = instance.text

                if instance.class_value == class_value:
                    trees += self.__nlp.parse(text)

                    binary_data.append(Instance(text, class_value))
                else:
                    binary_data.append(Instance(text, "Not" + class_value))

            num_words = self.__num_words[class_value] if \
                isinstance(self.__num_words, dict) and class_value in \
                self.__num_words else self.__num_words
            ig_words = self.__top_information_gain_words(binary_data,
                                                         num_words)

            # Extract patterns from dependency trees.
            for tree in trees:
                pattern_extractor.extract_patterns(ig_words, tree, class_value)

        self.__nlp.set_mode("evaluate")

        # Determine the weighted accuracy of patterns on training sentences.
        for instance in data:
            self.__nlp.test(instance.text, class_value)

        self.__nlp.set_mode("classify")
