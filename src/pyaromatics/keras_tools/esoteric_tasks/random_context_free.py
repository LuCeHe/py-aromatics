import nltk
from nltk.parse.generate import generate

import numpy as np

import string
import random

from pyaromatics.keras_tools.esoteric_tasks.nlp import Vocabulary

"""
  - set increasing difficulty (each of them 3 times: RELAX/REBAR/REINFORCE), (words:w, loops:l, states:w, ors:o)
    - 3o, 10s, 10w -> 0l, 1l, 2l
    - # [NOTO] 0 loops, 10 states  -> 10 words, 100 words, 1000 words
    - 3o, 1l, 10w  -> 10s, 50s, 100s
    - 10s, 1l, 10w -> 1o, 3o, 10o
"""

example_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP
    V -> "saw" | "ate" | "walked"
    NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
    Det -> "a" | "an" | "the" | "my"
    N -> "man" | "dog" | "cat" | "telescope" | "park"
    P -> "in" | "on" | "by" | "with"
    """)

reber_grammar = nltk.CFG.fromstring("""
    START -> 'B' 'T' REBER 'T' 'E' | 'B' 'P' REBER 'P' 'E'
    REBER -> 'B' E1
    E1 -> 'T' E2 | 'P' E3
    E2 -> 'S' E2 | 'X' E4
    E3 -> 'T' E3 | 'V' E5
    E4 -> 'X' E3 | 'S' E6
    E5 -> 'P' E4 | 'V' E6
    E6 -> 'E'
    """)


def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def RandomStates(number=3):
    states = [id_generator(size=2, chars=string.ascii_uppercase) for _ in range(number)]
    return sorted(states)


def RandomWords(number=6):
    words = ['\"{}\"'.format(id_generator(size=4, chars=string.ascii_lowercase)) for _ in range(number)]
    return sorted(words)


def RandomAlternatives_old(initial_state, states, words, max_number_ors_ands=3, no_loops=True, use_all_states=True):
    idx = states.index(initial_state)
    if no_loops:
        # to ensure the lack of loops the time direction of the directed graph is the alfabetical direction
        # it could be relaxed to have 1 time step loops, and n time steps loops
        states = states[idx + 1:]
        idx = -1

    number_ors_ands = np.random.choice(max_number_ors_ands)
    number_ors = np.random.choice(number_ors_ands) if not number_ors_ands == 0 else 0

    number_ands = number_ors_ands - number_ors
    n_sw = number_ors_ands + 1
    ws = np.random.choice(states + words, n_sw).tolist()

    if use_all_states:
        if idx < len(states) and len(states) > 0 and states[idx + 1] not in ws:
            ws = ws[:-1] + [states[idx + 1]]
            np.random.shuffle(ws)

    oa = [' | '] * number_ors + [' '] * number_ands
    np.random.shuffle(oa)

    alternatives = ''.join(['{}{}'.format(i, j) for i, j in zip(ws[:-1], oa)]) + ws[-1]
    return alternatives


def RandomAlternatives(initial_state, states, words, max_number_ors=3, no_loops=True, use_all_states=True):
    idx = states.index(initial_state)

    wxs = int(len(words) / len(states)) + 1
    splits = np.cumsum([wxs] * len(states)).tolist()
    split = np.split(np.array(words), splits)[:-1][idx]

    if no_loops:
        # to ensure the lack of loops the time direction of the directed graph is the alfabetical direction
        # it could be relaxed to have 1 time step loops, and n time steps loops
        states = states[idx + 1:]
        idx = -1

    number_ors_ands = np.random.choice(max_number_ors)
    number_ors = np.random.choice(max_number_ors) if not number_ors_ands == 0 else 0

    number_ands = number_ors_ands - number_ors
    n_sw = number_ors_ands + 1
    ws = np.random.choice(states + words, n_sw).tolist()

    if use_all_states:
        if idx < len(states) and len(states) > 0 and states[idx + 1] not in ws:
            ws = ws[:-1] + [states[idx + 1]]
            np.random.shuffle(ws)

    oa = [' | '] * number_ors + [' '] * number_ands
    np.random.shuffle(oa)

    alternatives = [None] * (len(ws) + len(oa))
    alternatives[::2] = ws
    alternatives[1::2] = oa

    # alternatives = ''.join(['{}{}'.format(i, j) for i, j in zip(ws[:-1], oa)]) + ws[-1]
    return alternatives


def RandomGrammar_old(number_states=4, number_words=2, max_number_ors=5):
    states = RandomStates(number=number_states)
    words = RandomWords(number=number_words)

    s_states = ['S'] + states

    list_alternatives = []
    for s in s_states:
        alternative = RandomAlternatives(s, s_states, words, max_number_ors=max_number_ors)
        print(alternative)
        list_alternatives.append(alternative)

    grammar_string = ''.join(['{} -> {}\n'.format(s, ''.join(a)) for s, a in zip(s_states, list_alternatives)])

    # check that all words are used
    grammar = nltk.CFG.fromstring(grammar_string)
    vocabulary = Vocabulary.fromGrammar(grammar)
    words_in_grammar = vocabulary.removeSpecialTokens()

    print()
    print('words_in_grammar: ', words_in_grammar)
    n_tokens = len(vocabulary.tokens)

    if n_tokens < number_words:
        words_left = [w for w in words if not w in words_in_grammar]
        print('words_left:       ', words_left)

    return grammar_string


def RandomStatesAdjacency(n_states, n_loops):
    # make sure the initial directed graph has no loops
    m = np.random.choice(2, (n_states, n_states))
    ut = np.triu(m, k=1)

    # make sure all the states can be visited
    d = np.diag(np.diag(np.ones((n_states, n_states)), k=1), k=1)
    adjacency = (ut + d > 0) * 1

    # add loops
    lt = np.tril(adjacency, k=0)
    while np.sum(lt) < n_loops:
        connect = sorted(np.random.choice(range(1, n_states), 2, replace=False).tolist())[::-1]
        adjacency[connect[0], connect[1]] = 1
        lt = np.tril(adjacency, k=0)

    return adjacency


def StatesPerRule(states, adjacency):
    n_states = len(states)
    m_adjacency = range(n_states) * adjacency
    list_states_to = []
    for i, s in enumerate(states):
        states_to = [states[st] for st in m_adjacency[i] if not st == 0]
        list_states_to.append(states_to)
    return list_states_to


def WordsPerRule(states, words):
    wxs = int(len(words) / len(states)) + 1
    splits = np.cumsum([wxs] * len(states)).tolist()
    splits = np.split(np.array(words), splits)[:-1]

    list_words_to = []
    # make sure every grammar rule has terminal states (words)
    for s in splits:
        rw = np.random.choice(words, 5)
        list_words_to.append(s.tolist() + rw.tolist())
    return list_words_to


def RandomGrammar(number_states=4, number_words=2, n_loops=0, n_ors=2):
    states = RandomStates(number=number_states)
    words = RandomWords(number=number_words)
    states = ['S'] + states

    adjacency = RandomStatesAdjacency(len(states), n_loops)

    list_spr = StatesPerRule(states, adjacency)
    list_wpr = WordsPerRule(states, words)

    rules = []
    for s, wt, st in zip(states, list_wpr, list_spr):
        ws = wt + st
        np.random.shuffle(ws)
        length_rule = len(ws)
        n_ands = length_rule - n_ors - 1
        oa = [' | '] * n_ors + [' '] * n_ands
        np.random.shuffle(oa)
        alternatives = [None] * (len(ws) + len(oa))
        alternatives[::2] = ws
        alternatives[1::2] = oa
        rules.append('{} -> {}\n'.format(s, ''.join(alternatives)))

    grammar_string = ''.join(rules)
    return grammar_string


if __name__ == '__main__':
    grammar_string = RandomGrammar(number_states=10, number_words=2, n_loops=0, n_ors=2)
    print()
    print(grammar_string)
    grammar = nltk.CFG.fromstring(grammar_string)
    sentences = set([' '.join(sentence) for sentence in generate(grammar, depth=10, n=20)])

    print()
    vocabulary = Vocabulary.fromGrammar(grammar)
    print(vocabulary.tokens)
    n_tokens = len(vocabulary.tokens) - 4

    print(n_tokens)
    print()
    for s in sentences:
        print(s)
