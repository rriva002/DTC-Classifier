import json
import math
import socket
from CNNClassifier import CNNClassifier
from collections import Counter
from extractPatterns import PatternExtractor
from nltk import word_tokenize
from RandomForestTextClassifier import RandomForestTextClassifier
from sklearn.feature_extraction.text import CountVectorizer
from TextClassifierUtils import AbstractTextClassifier, Instance


class NlpService(object):
    def __init__(self, host="localhost", port=9000):
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.__sock.connect((host, port))

    def __receive(self):
        data = ""

        while len(data) == 0 or not data[len(data) - 1] == "\n":
            data += self.__sock.recv(1).decode("latin-1")

        return json.loads(data[0:len(data) - 1].replace("__NEWLINE__", "\n"))

    def __send(self, command, data):
        data["command"] = command

        self.__sock.send((json.dumps(data) + "\n").encode())

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

    def set_split_sentences(self, split):
        self.__send("split_sentences", {"value": split})

    def test(self, text, class_value):
        self.__send("test", {"text": text, "class": class_value})


class DefaultRuleClassifier(AbstractTextClassifier):
    def __init__(self):
        self.__default_distribution = {}

    def classify(self, data):
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
    def __init__(self, backup_classifier=None, nlp_host="localhost",
                 num_words=10, max_words=4):
        self.__backup_classifier = DefaultRuleClassifier() if \
            backup_classifier is None else backup_classifier
        self.__nlp = NlpService(host=nlp_host)
        self.__num_words = num_words
        self.__max_words = max(1, max_words)

    def __entropy(self, vectors, index):
        value_frequencies = {}
        entropy = 0

        for vector in vectors:
            if vector[index] not in value_frequencies:
                value_frequencies[vector[index]] = 0

            value_frequencies[vector[index]] += 1

        for frequency in value_frequencies.values():
            ratio = frequency / len(vectors)
            entropy -= ratio * math.log(ratio, 2)

        return entropy

    def __stem_text(self, text):
        if self.__stemmer is None:
            return text

        stemmed_text = ""

        for word in word_tokenize(text):
            stemmed_text += (" " if len(stemmed_text) > 0 else
                             "") + self.__stemmer.stem(word)

        return stemmed_text

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
        distribution = self.__nlp.classify(self.__stem_text(instance.text),
                                           instance.class_value)
        distribution = self._normalize_distribution(distribution)

        return distribution if len(distribution) > 0 else \
            self.__backup_classifier.classify(instance)

    def disconnect(self):
        if self.__nlp is not None:
            self.__nlp.end()

    def train(self, data):
        classes = []

        for instance in data:
            classes.append(instance.class_value)

        counter = Counter(classes)
        most_common = counter.most_common(1)[0][0]
        classes = set(classes)
        num_classes = len(classes)

        for class_value in classes:
            print(class_value + ": " + str(counter[class_value]))

        if self.__ppdb is not None:
            data = data + self.__ppdb.balance(data)

        self.__backup_classifier.train(data)

        pattern_extractor = PatternExtractor(self.__nlp, self.__max_words)

        self.__nlp.set_mode("train")
        self.__nlp.set_split_sentences(self.__split_sentences)

        if(counter[most_common] / len(data) > self.__imbalance_threshold):
            classes.remove(most_common)

        if num_classes > 2:
            for class_value in classes:
                binary_data = []
                trees = []

                for instance in data:
                    text = self.__stem_text(instance.text)

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

                for tree in trees:
                    pattern_extractor.extract_patterns(ig_words, tree,
                                                       class_value)

        self.__nlp.set_mode("evaluate")

        for instance in data:
            self.__nlp.test(instance.text, class_value)

        self.__nlp.set_mode("classify")
