import nltk
from nltk.parse.generate import generate

import numpy as np

import string
import random


def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


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


def RandomStates(number=3):
    states = [id_generator(size=2, chars=string.ascii_uppercase) for _ in range(number)]
    return sorted(states)


def RandomWords(number=6):
    words = ['\"{}\"'.format(id_generator(size=4, chars=string.ascii_lowercase)) for _ in range(number)]
    return sorted(words)

def RandomAlternatives(initial_state, states, words, max_number_ors_ands=3, no_loops=True, use_all_states=True):
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
        if idx < len(states) and len(states) > 0 and states[idx+1] not in ws:
            ws = ws[:-1] + [states[idx+1]]
            np.random.shuffle(ws)

    oa = [' | '] * number_ors + [' '] * number_ands
    np.random.shuffle(oa)

    alternatives = ''.join(['{}{}'.format(i, j) for i, j in zip(ws[:-1], oa)]) + ws[-1]
    return alternatives

def RandomGrammar(number_states=4, number_words=2, max_number_ors_ands=5):
    states = RandomStates(number=number_states)
    words = RandomWords(number=number_words)

    s_states = ['S'] + states
    grammar_string = ''.join(['{} -> {}\n'.format(s, RandomAlternatives(s, s_states, words, max_number_ors_ands=max_number_ors_ands)) for s in s_states])

    return grammar_string


if __name__ == '__main__':
    grammar_string = RandomGrammar(number_states=4, number_words=4, max_number_ors_ands=5)
    print(grammar_string)
    grammar = nltk.CFG.fromstring(grammar_string)
    sentences = set([' '.join(sentence) for sentence in generate(grammar, depth=10)])

    print()
    for s in sentences:
        print(s)