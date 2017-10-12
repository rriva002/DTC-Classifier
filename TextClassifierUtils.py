import csv
from abc import ABC, abstractmethod
from random import random


class AbstractTextClassifier(ABC):
    """Abstract class for text classifiers."""
    @abstractmethod
    def train(self, data):
        """Train the classifier on the given training set.

        Arguments:
        data - A list of Instance objects (defined in TextDataSetFileParser.py)

        Returns:
        Nothing
        """
        pass

    @abstractmethod
    def classify(self, instance):
        """Determine the probability distribution of the given instance.

        Arguments:
        instance - An instance object (defined in TextDataSetFileParser.py)

        Returns:
        A dictionary with class strings as keys and probabilities as values
        """
        pass

    def evaluate(self, test_set, verbose=False, output_matrix=None):
        """Evaluate the classifier's performance on the given test set.

        Arguments:
        test_set - A list of Instance objects (defined in
        TextDataSetFileParser.py)
        verbose (default False) - If True, print the results of the evaluation
        output_matrix (default None) - A dictionary that is populated with
        an X marking each incorrectly classified instance (second key) for each
        class (first key) for each classifier in the ensemble.

        Returns:
        A dictionary with the following key-value pairs:
            accuracy - The ratio of correctly classified instances
            weightedaccuracy - The ratio of weights of correctly classified
            instances
            confusionmatrix - A "two-dimensional dictionary" where matrix[A][B]
            yields the number of instances of class A that were classified
            as class B by the classifier
        """
        correct = 0
        weighted_correct = 0
        weighted_total = 0
        confusion_matrix = {}
        column_width = {}
        classes = set()

        for instance in test_set:
            max_class = None
            max_probability = -1
            weighted_total += instance.weight

            for class_value, probability in self.classify(instance).items():
                if (probability > max_probability or
                        probability == max_probability and random() < 0.5):
                    max_class = class_value
                    max_probability = probability

                if class_value not in confusion_matrix:
                    confusion_matrix[class_value] = {}

                    for c_val in classes:
                        confusion_matrix[class_value][c_val] = 0

                    classes.add(class_value)

                    for c_val in classes:
                        confusion_matrix[c_val][class_value] = 0

            if max_class == instance.class_value:
                correct += 1
                weighted_correct += instance.weight

                if output_matrix is not None:
                    if max_class not in output_matrix:
                        output_matrix[max_class] = {}

                    if instance.text not in output_matrix[max_class]:
                        output_matrix[max_class][instance.text] = []

                    output_matrix[max_class][instance.text].append("")
            elif output_matrix is not None:
                if instance.class_value not in output_matrix:
                    output_matrix[instance.class_value] = {}

                if instance.text not in output_matrix[instance.class_value]:
                    output_matrix[instance.class_value][instance.text] = []

                output_matrix[instance.class_value][instance.text].append("X")

            if instance.class_value not in confusion_matrix:
                confusion_matrix[instance.class_value] = {}

                for class_value in classes:
                    confusion_matrix[instance.class_value][class_value] = 0

                classes.add(instance.class_value)

                for class_value in classes:
                    confusion_matrix[class_value][instance.class_value] = 0

            confusion_matrix[instance.class_value][max_class] += 1

            if verbose and instance.class_value not in column_width:
                column_width[instance.class_value] = len(instance.class_value)

        accuracy = correct / len(test_set)
        sum_accuracies = 0.0
        adjustment = 0.0

        for c1 in confusion_matrix:
            TC = confusion_matrix[c1][c1]
            C = 0.0

            for c2 in confusion_matrix[c1]:
                C += confusion_matrix[c1][c2]

            if C > 0.0:
                sum_accuracies += TC / C
            else:
                adjustment += 1

        weighted_acc = sum_accuracies / (len(confusion_matrix) - adjustment)

        if verbose:
            classes = list(classes)

            classes.sort()
            print(("Accuracy: %0.2f" % (100 * accuracy)) + "%")
            print(("Weighted Accuracy: %0.2f" % (100 * weighted_acc)) + "%")
            print("Confusion Matrix:")

            for class_value, distribution in confusion_matrix.items():
                for prediction, count in distribution.items():
                    if prediction not in column_width:
                        width = max(len(prediction), len(str(count)))
                        column_width[prediction] = width
                    elif prediction in column_width:
                        if len(str(count)) > column_width[prediction]:
                            column_width[prediction] = len(str(count))

            for class_value in classes:
                row = ""

                for prediction in classes:
                    width = column_width[prediction] - len(str(prediction)) + 1

                    for i in range(0, width):
                        row += " "

                        row += prediction

                print(row + " <- Classified As")
                break

            for class_value in classes:
                row = ""

                for prediction in classes:
                    str_val = str(confusion_matrix[class_value][prediction])
                    width = column_width[prediction] - len(str_val) + 1

                    for i in range(0, width):
                        row += " "

                    row += str(confusion_matrix[class_value][prediction])

                print(row + " " + class_value)

        return {"accuracy": accuracy, "weightedaccuracy": weighted_acc,
                "confusionmatrix": confusion_matrix}

    def _normalize_distribution(self, distribution):
        sum_of_probabilities = 0

        for class_value, probability in distribution.items():
            sum_of_probabilities += probability

        if sum_of_probabilities > 0:
            for class_value, probability in distribution.items():
                distribution[class_value] = probability / sum_of_probabilities

        return distribution


class Instance(object):
    """Container for a single text data instance.

    Constructor arguments:
    text - Text data string
    class_value - The class of the text data
    weight (default 1) - The weight of the text data
    """
    def __init__(self, text, class_value, weight=1):
        self.text = text
        self.class_value = class_value
        self.weight = weight


class TextDatasetFileParser(object):
    """Reader for text dataset files.

    Constructor arguments:
    verbose (default False) - If True, print each line of the file as it's read
    """
    def __init__(self, verbose=False):
        self.__verbose = verbose

    def parse(self, filename, delimiter=",", quotechar='"'):
        """Read an ARFF or CSV file containing a dataset. ARFF files should be
        formatted according to Weka's standards. CSV files should be
        in "text,class,weight" format (weight is optional).

        Arguments:
        filename - The path of the file to read

        Returns:
        A list of Instance objects made from the data contained in the file
        """
        if filename.endswith(".arff"):
            return self.__parse_arff_file(filename)
        elif filename.endswith(".csv"):
            return self.__parse_csv_file(filename, delimiter, quotechar)

        return []

    def parse_unlabeled(self, filename):
        """Read a text file containing unlabeled text data. The file should
        have one text data string per line.

        Arguments:
        filename - The path of the file to read

        Returns:
        A list of strings made from the data contained in the file
        """
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            for line in file:
                dataset.append(line.lower())

        return dataset

    def __parse_arff_file(self, filename):
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            parsing_data = False
            attributes = []

            for line in file:
                line = line.strip()

                if len(line) == 0:
                    continue
                elif not parsing_data and \
                        line.upper().startswith("@ATTRIBUTE"):
                    if line.find("{") >= 0:
                        data_type = "NOMINAL"
                    else:
                        data_type = line[line.rfind(" ") + 1:].upper()

                    if self.__verbose:
                        print("Attribute: " + data_type)

                    attributes.append(data_type)
                elif not parsing_data and line.upper() == "@DATA":
                    parsing_data = True
                elif parsing_data:
                    text = ""
                    value = ""
                    weight = 1
                    in_quotes = False

                    if self.__verbose:
                        print(line)

                    if line.endswith("}"):
                        index = line.rfind(",{")

                        if index >= 0:
                            weight = float(line[index + 2:len(line) - 1])
                            line = line[:index]

                    index = line.rfind(",")
                    label = line[index + 1:]
                    line = line[:index]

                    for i in range(0, len(line)):
                        if line[i] == "'" and (i == 0 or line[i - 1] != "\\"):
                            in_quotes = not in_quotes
                        elif not in_quotes and line[i] == ",":
                            text += (" " if len(text) > 0 else "") + value

                            value = ""
                        elif line[i] != "\\":
                            value += line[i]

                    dataset.append(Instance(text, label, weight))

        return dataset

    def __parse_csv_file(self, filename, delimit, quote_char):
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            reader = csv.reader(file, delimiter=delimit, quotechar=quote_char)

            for line in reader:
                if self.__verbose:
                    print(line)

                if len(line) > 1:
                    class_value = line[1]
                else:
                    class_value = None

                if len(line) > 2:
                    weight = float(line[2])
                else:
                    weight = 1

                dataset.append(Instance(line[0], class_value, weight))

        return dataset
