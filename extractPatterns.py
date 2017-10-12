from itertools import combinations


class Node(object):
    def __init__(self, node_val):
        self.node_val = node_val
        self.children = []
        self.parents = []

    def __repr__(self):
        word1, pos = self.node_val
        word = {} if word1 == "*" else "{word:/%s.*/}" % word1

        if len(self.children) == 0:
            return word

        child_str = ""

        for rel, child in self.children:
            if rel == "indir":
                child_str += " >> %s" % child
            else:
                child_str += " > %s" % child if rel != "neg" else " >neg {}"

        return "(%s %s)" % (word, child_str)

    def add_child(self, rel_child):
        rel, child = rel_child
        self.children += [rel_child]  # tuples of the form (rel, node)
        child.parents += [(rel, self)]   # tuples of the form (rel, node)

    def change_word_to_star(self):
        self.node_val = ("*", "*")
        return self

    def clone(self):
        node_val_clone = self.node_val
        n = Node(node_val_clone)

        for rel, child in self.children:
            cloned_child = child.clone()
            cloned_child.parents += [(rel, n)]
            n.children += [(rel, cloned_child)]

        return n

    def get_all_important_word_nodes(self, word_nodes):
        word, po = self.node_val
        is_neg_node = False

        if len(self.parents) > 0:  # is not root node
            rel, parent = self.parents[0]

            if rel == "neg":
                is_neg_node = True

        if word != "*" and not is_neg_node:
            # then this is an important word and we add it to the dict
            if word not in word_nodes:
                word_nodes[word] = []

            word_nodes[word] += [self]

        for rel, child in self.children:
            # regardless of whether this node is important
            # or not, we consider its children
            child.get_all_important_word_nodes(word_nodes)


class Tree(object):
    def __init__(self, root):
        self.__root = root

    def __repr__(self):
        return self.__root.__repr__()

    def get_root(self):
        return self.__root


class PatternExtractor(object):
    def __init__(self, nlp, max_words):
        self.__nlp = nlp
        self.__max_words = max_words

    def __add_suffix_to_all_non_star_nodes(self, node, i_list):
        i = i_list[0]
        i_list[0] = i + 1
        word, pos = node.node_val

        if word != "*":
            node.node_val = ("%s__%s" % (word, i), pos)

        for rel, child in node.children:
            self.__add_suffix_to_all_non_star_nodes(child, i_list)

    def __consume_next_token(self, s, position):
        if position == len(s):
            return None

        start = position
        tokens_a = [" ", "\t", "\n", "\r"]

        while start < len(s) and s[start] in tokens_a:
            start += 1

        if position >= len(s):
            return None

        end = start
        tokens_b = ["[", "]", ">", "/"]

        if s[start] in tokens_b:
            return (s[start], start + 1)

        while end < len(s) and s[end] not in tokens_a + tokens_b:
            end += 1

        return (s[start:end].strip(), end)

    def __create_all_patterns_and_sub_patterns(self, node1, important_words,
                                               class_value):
        if len(node1.children) == 0:
            return

        node = node1.clone()
        # prunes the tree to keep only branches that contain an important word.
        # non-important words along the way are replaced by "*". For negation,
        # we don't care about the word; instead we only consider the relation
        # "neg", so wee keep branches containing a "neg" relation too.
        sub_tree = self.__get_sub_tree(node, important_words)

        if sub_tree is not None:
            if self.__nlp.has_pattern(self.__pattern_to_str(sub_tree),
                                      class_value):
                return

            sub_tree2 = self.__prune_empty_roots(sub_tree)

            self.__make_indir_relations(sub_tree2)

            if len(sub_tree2.children) == 0:
                return

            pattern = self.__pattern_to_str(sub_tree2)

            if self.__nlp.has_pattern(pattern, class_value):
                return

            self.__nlp.add_pattern(pattern, class_value)
            self.__create_sub_patterns(sub_tree2, class_value)

    def __create_sub_patterns(self, pattern_tree, class_value):
        important_word_nodes = {}

        # this will fill word_nodes with data
        pattern_tree.get_all_important_word_nodes(important_word_nodes)

        important = important_word_nodes.keys()

        if len(important) <= 2:
            # if patternTree has only 2 important words,
            # then we don't create subpatterns for it
            return

        combos = self.__create_word_combinations(important)

        for word_comb in combos:
            important = word_comb

            self.__create_all_patterns_and_sub_patterns(pattern_tree,
                                                        important, class_value)

    def __create_word_combinations(self, words):
        return_list = []

        for word_to_remove in words:
            comb = []

            for word in words:
                comb += [word]

            comb.remove(word_to_remove)

            return_list += [comb]

        return return_list

    def __get_sub_tree(self, node, important_words):
        if len(node.children) > 0:
            good_children = []

            for rel, child in node.children:
                good_child = self.__get_sub_tree(child, important_words)

                if good_child is not None:
                    good_children += [(rel, child)]

            node.children = good_children

        if len(node.children) == 0:
            if not self.__is_important_node(node, important_words):
                # doesn't have important children and is not important itself
                return None

            # doesn't have important children, but is important itself
            return node

        if not self.__is_important_node(node, important_words):
            # has important children but is not important itself
            return node.change_word_to_star()

        return node  # has important children and is important itself

    def __is_important_node(self, node, important_words):
        if len(node.parents) > 0:
            rel, parent = node.parents[0]

            if rel == "neg":
                return True

        word, po = node.node_val

        for lemma in important_words:
            if lemma == word[:len(lemma)]:
                node.node_val = (lemma, po)  # replace word for lemma in node
                return True

        return False

    def __is_star(self, node):
        word, pos = node.node_val

        if word == "*" and pos == "*":
            return True

        return False

    # if a star node s with parent p has only one child c,
    # make c a grandchild of p with indirect relation
    def __make_indir_relations(self, n):
        new_children = []

        for rel, child in n.children:
            was_replaced = False

            while self.__is_star(child) and len(child.children) == 1:
                (rel, grand_child) = child.children[0]
                child = grand_child
                was_replaced = True

            if was_replaced:
                rel = "indir"
            new_children += [(rel, child)]
        n.children = new_children
        for rel, child in n.children:
            self.__make_indir_relations(child)

    def __pattern_to_str(self, pattern):
        pattern = pattern.clone()

        self.__remove_suffixes(pattern)

        pattern = str(pattern)

        if pattern[0] == "(":
            pattern = pattern[1:-1]

        return pattern.replace("  ", " ")

    def __peek_next_token(self, s, position):
        if position == len(s):
            return None

        start = position
        tokens_a = [" ", "\t", "\n", "\r"]

        while start < len(s) and s[start] in tokens_a:
            start += 1

        if position >= len(s):
            return None

        end = start
        tokens_b = ["[", "]", ">", "/"]

        if s[start] in tokens_b:
            return s[start]

        while end < len(s) and s[end] not in tokens_a + tokens_b:
            end += 1

        return s[start:end].strip()

    def __prune_empty_roots(self, node):
        while self.__is_star(node) and len(node.children) == 1:
            rel, child = node.children[0]
            node = child

        return node

    def __read_node(self, s, position):
        has_children = False
        next_token = self.__peek_next_token(s, position)

        if next_token == "[":  # the node we are going to read has children
            token, position = self.__consume_next_token(s, position)
            has_children = True

        node_val, position = self.__read_node_val(s, position)
        node = Node(node_val)

        if not has_children:
            return (node, position)

        while True:
            if self.__peek_next_token(s, position) == "]":
                token, position = self.__consume_next_token(s, position)
                break

            rel, position = self.__read_relation(s, position)
            child, position = self.__read_node(s, position)

            node.add_child((rel, child))

        return (node, position)

    def __read_node_val(self, s, position):
        position1 = position
        word, position = self.__consume_next_token(s, position)
        token, position = self.__consume_next_token(s, position)

        if token != "/":
            print("!!! warning: wrong token separating word/pos. Token is " +
                  "%s at position %s in str %s with initial position %s" %
                  (token, position, s, position1))

        pos, position = self.__consume_next_token(s, position)
        return ((word, pos), position)

    def __read_relation(self, s, position):
        start = position
        end = start

        while end < len(s) and s[end] != ">":
            end += 1

        if s[end] != ">":
            print("!! warning: > is missing from relation!")

        return (s[start:end].strip(), end + 1)

    def __read_tree(self, s):
        root, position = self.__read_node(s, 0)
        return Tree(root)

    def __remove_suffixes(self, node):
        word, pos = node.node_val
        index = word.find("__")

        if index > 0:
            word = word[:index]
            node.node_val = (word, pos)

        for rel, child in node.children:
            self.__remove_suffixes(child)

    def extract_patterns(self, important_words, tree, class_value):
        self.current_tree = tree
        # we clone the original tree, because we are going to change the
        # clone in place
        root_clone = self.__read_tree(tree).get_root().clone()
        # initial prune of the parse tree to obtain only nodes that are
        # members of important_words
        sub_tree = self.__get_sub_tree(root_clone, important_words)

        if sub_tree is None or len(sub_tree.children) == 0:
            # no important word found, or there is only one node in subtree
            return

        # now pattern_tree's words are replaced by word_suffix and hence
        # repeated words become unique. suffx is __i where i is a number
        # counting up from 1
        # self.__add_suffix_to_all_non_star_nodes(sub_tree, [0])

        important_word_nodes = {}

        # this will fill word_nodes with data. We want to extract all
        # important words (plus suffixes) that occur in this tree.
        sub_tree.get_all_important_word_nodes(important_word_nodes)

        # words_suffixes are important words plus suffixes.
        words_suffixes = important_word_nodes.keys()

        # this will fill allPatterns
        for subset in combinations(words_suffixes, min(len(words_suffixes),
                                                       self.__max_words)):
            self.__create_all_patterns_and_sub_patterns(sub_tree, subset,
                                                        class_value)

    def extract_patterns_from_trees(self, important_words, trees, class_value):
        for tree in trees:
            self.extract_patterns(important_words, tree, class_value)
