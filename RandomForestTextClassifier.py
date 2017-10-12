import numpy as np
from nltk.stem.snowball import EnglishStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from TextClassifierUtils import AbstractTextClassifier


class RandomForestTextClassifier(AbstractTextClassifier):
    """Random forest classifier that converts text to the bag-of-words model
    before training/classification. For further information, including
    constructor arguments, see scikit-learn's random forest documentation.
    """
    def __init__(self, num_trees=2000, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features="auto",
                 max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True,
                 oob_score=False, num_jobs=1, random_state=None, verbose=0,
                 warm_start=False, class_weight=None, ngram_range=(1, 3),
                 min_df=0.03, max_word_features=1000, tf_idf=True):
        stemmer = EnglishStemmer()

        if tf_idf:
            analyzer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                                       max_features=max_word_features,
                                       stop_words="english").build_analyzer()
        else:
            analyzer = CountVectorizer(ngram_range=ngram_range, min_df=min_df,
                                       max_features=max_word_features,
                                       stop_words="english").build_analyzer()

        def stemmed_words(text):
            return (stemmer.stem(word) for word in analyzer(text))

        if tf_idf:
            self.__vectorizer = TfidfVectorizer(ngram_range=ngram_range,
                                                min_df=min_df,
                                                max_features=max_word_features,
                                                stop_words="english",
                                                analyzer=stemmed_words)
        else:
            self.__vectorizer = CountVectorizer(ngram_range=ngram_range,
                                                min_df=min_df,
                                                max_features=max_word_features,
                                                stop_words="english",
                                                analyzer=stemmed_words)
        self.__random_forest = RandomForestClassifier(num_trees, criterion,
                                                      max_depth,
                                                      min_samples_split,
                                                      min_samples_leaf,
                                                      min_weight_fraction_leaf,
                                                      max_features,
                                                      max_leaf_nodes,
                                                      min_impurity_split,
                                                      bootstrap, oob_score,
                                                      num_jobs, random_state,
                                                      verbose, warm_start,
                                                      class_weight)

    def train(self, data):
        training_data = []
        training_labels = []
        training_weights = []

        for instance in data:
            training_data.append(instance.text)
            training_labels.append(instance.class_value)
            training_weights.append(instance.weight)

        training_data = self.__vectorizer.fit_transform(training_data)

        self.__random_forest.fit(training_data, np.array(training_labels),
                                 np.array(training_weights))

    def classify(self, instance):
        distribution = {}
        test_data = self.__vectorizer.transform([instance.text])
        ordered_dist = self.__random_forest.predict_proba(test_data)

        for i in range(0, len(ordered_dist[0])):
            if ordered_dist[0, i] > 0:
                class_value = self.__random_forest.classes_[i]
                distribution[class_value] = ordered_dist[0, i]

        return distribution
