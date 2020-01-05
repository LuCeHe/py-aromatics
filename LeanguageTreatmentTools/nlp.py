# Copyright (c) 2018, 
#
# authors: Simon Brodeur, Luca Celotti
# during their PhD at Universite' de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import gzip
import json
import logging
import random
import sys
import re
from enum import Enum

import nltk
import numpy as np
import six
from nltk.grammar import Nonterminal, CFG, Production

logger = logging.getLogger(__name__)


class Parser(object):

    def parse(self, sentence):
        raise NotImplementedError()


class Interpreter(object):

    def parse(self, tree):
        raise NotImplementedError()


class NltkParser(Parser):

    def __init__(self, grammar):
        if not isinstance(grammar, nltk.CFG):
            logger.info('Using grammar file: %s' % (grammar))
            grammar = nltk.data.load('file:' + grammar)
        parser = nltk.ChartParser(grammar)

        self.__dict__.update(grammar=grammar, parser=parser)

    def parse(self, sentence):
        tokens = tokenize(sentence)
        trees = [tree for tree in self.parser.parse(tokens)]
        return trees


class NltkGrammarSampler(object):

    def __init__(self, grammar, outputTokens=False):

        if not isinstance(grammar, nltk.CFG):
            logger.info('Using grammar file: %s' % (grammar))
            grammar = nltk.data.load('file:' + grammar)

        self.__dict__.update(grammar=grammar, outputTokens=outputTokens)

    def generate(self, n, start=None, depth=None):
        if not start:
            start = self.grammar.start()
        if depth is None:
            depth = sys.maxsize

        sentences = []
        for _ in range(n):
            tokens = self._generate_random([start], depth)
            if self.outputTokens:
                sentence = tokens
            else:
                sentence = ' '.join(tokens)
            sentences.append(sentence)

        return sentences

    def _generate_random(self, items, depth):

        tokens = []
        try:
            for item in items:
                if isinstance(item, Nonterminal):
                    if depth > 0:
                        prods = self.grammar.productions(lhs=item)
                        if len(prods) > 0:
                            prod = random.choice(prods)
                            tokens.extend(self._generate_random(
                                prod.rhs(), depth - 1))
                else:
                    tokens.append(item)

        except RuntimeError as _error:
            if _error.message == "maximum recursion depth exceeded":
                raise RuntimeError(
                    "The grammar has rule(s) that yield infinite recursion!!")
            else:
                raise

        return tokens


def tokenize(sentence):
    sentence = sentence.replace('  ', ' ')
    tokens = sentence.split(' ')

    if len(tokens) == 1 and tokens[0] == '':
        tokens = ['the']
    if tokens[0] == '':
        tokens = tokens[1:]
    if tokens[-1] == '':
        tokens = tokens[:-1]
    return tokens


def mergeGrammars(grammars):
    mergedStart = Nonterminal('S')
    productions = []
    for i, grammar in enumerate(grammars):
        start = Nonterminal(grammar.start().symbol() + '-' + str(i))
        productions.append(Production(mergedStart, [start]))

        for production in grammar.productions():
            lhs = Nonterminal(production.lhs().symbol() + '-' + str(i))
            rhs = []
            for node in production.rhs():
                if isinstance(node, Nonterminal):
                    rhs.append(Nonterminal(node.symbol() + '-' + str(i)))
                else:
                    rhs.append(node)
            productions.append(Production(lhs, rhs))

    return CFG(mergedStart, productions)


def addEndTokenToGrammar(grammar, endToken):
    newStart = Nonterminal('S')
    productions = []

    i = 0
    start = Nonterminal(grammar.start().symbol() + '-' + str(i))
    productions.append(Production(newStart, [start, endToken]))

    for production in grammar.productions():
        lhs = Nonterminal(production.lhs().symbol() + '-' + str(i))
        rhs = []
        for node in production.rhs():
            if isinstance(node, Nonterminal):
                rhs.append(Nonterminal(node.symbol() + '-' + str(i)))
            else:
                rhs.append(node)
        productions.append(Production(lhs, rhs))

    return CFG(newStart, productions)


class Vocabulary(object):
    padToken = '<PAD>'
    startToken = '<START>'
    endToken = '<END>'
    unkToken = '<UNK>'
    specialTokens = [padToken, startToken, endToken, unkToken]

    def __init__(self, tokens, grammar=None):

        if Vocabulary.endToken in tokens:
            tokens.remove(Vocabulary.endToken)

        indicesByTokens = dict()
        tokens = Vocabulary.specialTokens + sorted(list(tokens))
        for i, token in enumerate(tokens):
            indicesByTokens[token] = i
        self.__dict__.update(tokens=tokens,
                             indicesByTokens=indicesByTokens,
                             grammar=grammar)

        self.padIndex = indicesByTokens[self.padToken]
        self.startIndex = indicesByTokens[self.startToken]
        self.endIndex = indicesByTokens[self.endToken]
        self.unkIndex = indicesByTokens[self.unkToken]

    def __eq__(self, other):
        return self.tokens == other.tokens

    def __ne__(self, other):
        return self.tokens != other.tokens

    def __add__(self, other):
        # NOTE: ignore the end token
        tokens = set(self.tokens[1:])
        tokens.update(other.tokens[1:])
        return Vocabulary(list(sorted(tokens)))

    def sort(self):
        # NOTE: ignore the end token
        tokens = sorted(self.tokens[1:])
        self.indicesByTokens = dict()
        self.tokens = [Vocabulary.endToken] + tokens
        for i, token in enumerate(self.tokens):
            self.indicesByTokens[token] = i

    def indexToToken(self, idx):
        return self.tokens[idx]

    def indicesToTokens(self, indices, offset=0):
        return [self.tokens[i - offset] for i in indices]

    def tokenToIndex(self, token, offset=0):
        return self.indicesByTokens[token] + offset

    def tokensToIndices(self, tokens, offset=0):
        indices = []
        for token in tokens:
            indices.append(self.indicesByTokens[token] + offset)
        return indices

    def sentenceToIndices(self, sentence, offset=0):
        return self.tokensToIndices(tokenize(sentence), offset)

    def sentenceToTokens(self, sentence):
        return tokenize(sentence)

    def sentencesToIndices(self, sentences, offset=0):
        indices = [self.sentenceToIndices(sentence, offset) for sentence in sentences]
        return indices

    def tokensToSentence(self, tokens):
        return ' '.join(tokens)

    def indicesToSentence(self, indices, offset=0):
        return ' '.join(self.indicesToTokens(indices, offset))

    def indicesToSentences(self, indices_list, offset=0):
        if type(indices_list).__module__ == 'numpy':
            indices_list = indices_list.tolist()
            # unpad:
            indices_list = [list(filter((0).__ne__, indices)) for indices in indices_list]

        sentences = [self.indicesToSentence(indices, offset) for indices in indices_list]
        return sentences

    def removeSpecialTokens(self, tokens):
        return [token for token in tokens if token not in self.specialTokens]

    def fromStartToEnd(self, tokens):
        try:
            start_location = tokens.index(self.startToken)
            end_location = tokens.index(self.endToken)
            tokens = tokens[start_location + 1:end_location]
        except:
            pass
        try:
            end_location = tokens.index(self.endToken)
            tokens = tokens[:end_location]
        except:
            pass
        try:
            tokens = self.removeSpecialTokens(tokens)
            q_location = tokens.index('?')
            tokens = tokens[:q_location + 1]
        except:
            pass
        return tokens

    def toFile(self, filename):
        with open(filename, 'w') as f:
            data = {'tokens': self.tokens}
            json.dump(data, f,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)

    @staticmethod
    def fromGrammar(grammar):
        tokens = []
        for production in grammar.productions():
            for p in production.rhs():
                if not isinstance(p, Nonterminal):
                    tokens.append(p)

        # Remove redundant tokens and sort
        tokens = list(set(tokens))
        tokens.sort()

        return Vocabulary(tokens=tokens, grammar=grammar)

    @staticmethod
    def fromGrammarFile(grammarCfg):
        grammar = nltk.data.load('file:' + grammarCfg)
        return Vocabulary.fromGrammar(grammar)

    @staticmethod
    def fromFile(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            tokens = data['tokens']
        return Vocabulary(tokens)

    @staticmethod
    def fromGz(data_filepaths):
        if not isinstance(data_filepaths, list): data_filepaths = [data_filepaths]
        tokens = []
        for filepath in data_filepaths:
            f = gzip.open(filepath, 'rb')
            for line in f:
                sentence = line.decode('windows-1252').strip()
                some_tokens = tokenize(sentence)

                tokens.extend(some_tokens)

        tokens = list(set(tokens))
        return Vocabulary(tokens)

    def getMaxVocabularySize(self):
        return len(self.tokens)

def preprocessSentence(sentence):
    s = sentence
    s = re.sub('([.,/-:<>\!?{}()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    sentence = s
    sentence = sentence.lower()

    sentence = sentence.replace('-', ' - ')
    sentence = sentence.replace("\"", "'")
    sentence = sentence.replace("'", " ' ")
    sentence = sentence.replace('  ', ' ')
    return sentence

def postprocessSentence(sentence):
    tokens = tokenize(sentence)

    # Fix determiners
    vowel = 'a', 'e', 'i', 'o', 'u'
    for i in range(len(tokens)):
        if i == len(tokens) - 1:
            continue

        if tokens[i] == 'a':
            if tokens[i + 1].startswith(vowel):
                tokens[i] = 'an'
        elif tokens[i] == 'an':
            if not tokens[i + 1].startswith(vowel):
                tokens[i] = 'a'

    sentence = ' '.join(tokens)

    return sentence


def getApproximateMaxSentenceLengthFromGrammar(grammarCfg, n=100000):
    sampler = NltkGrammarSampler(grammarCfg, outputTokens=True)
    sentences = sampler.generate(n)
    maxTokens = len(max(sentences, key=len))
    return maxTokens


class NodeType(Enum):
    undefined = 1
    terminal = 2
    nonterminal = 3
    terminalgroup = 4


class Node(object):

    def __init__(self, name, nodeType, parent=None):
        self.__dict__.update(name=name, parent=parent, nodeType=nodeType,
                             children=[], next_sibbling=None)

    def __repr__(self):
        return '(%s, type=%s)' % (str(self.name), str(self.nodeType.name))

    def __str__(self):
        return self.__repr__()

    def pformat(self, margin=70, indent=0, nodesep='', parens='()', quotes=False):
        s = '%s%s%s' % (parens[0], str(self.name), nodesep)
        for child in self.children:
            s += '\n' + ' ' * (indent + 2) + child.pformat(margin,
                                                           indent + 2, nodesep, parens, quotes)
        return s + parens[1]

    def getPathFromRoot(self):
        path = []
        node = self
        while node.parent is not None:
            parent = node.parent
            path.append(parent)
            node = parent
        path = path[::-1]
        path.append(self)
        return path

    def findNextSibbling(self):
        if self.next_sibbling is not None:
            # Find immediate next sibbling node
            sibbling = self.next_sibbling
        elif self.parent is not None:
            # Find first next sibbling node when searching bottom-up in the
            # tree
            sibbling = self.parent.findNextSibbling()
        else:
            sibbling = None
        return sibbling


class NonTerminalNode(Node):

    def __init__(self, name, parent=None):
        super(NonTerminalNode, self).__init__(
            name, NodeType.nonterminal, parent)


class TerminalNode(Node):

    def __init__(self, name, token, parent=None):
        Node.__init__(self, name, NodeType.terminal, parent)
        self.__dict__.update(token=token)

    def match(self, token):
        return self.token == token


class TerminalGroupNode(Node):

    def __init__(self, name, startTokens, productions, parent=None):
        Node.__init__(self, name, NodeType.terminalgroup, parent)
        self.__dict__.update(startTokens=startTokens, productions=productions, expandedStartTokens=set())

    def match(self, token):
        return token in self.startTokens

    def expandForMultitoken(self, startToken):
        mwStartNodes = []
        if startToken in self.expandedStartTokens:
            # Find existing expanded node with the same start token
            for child in self.children:
                if child.token == startToken:
                    mwStartNodes.append(child)
            if len(mwStartNodes) == 0:
                raise Exception('Unable to find existing expanded node!' + '\n' + str(self) + ' : ' + str(startToken))
        else:
            # Expand with the provided start token
            for production in self.productions:
                rhs = production.rhs()

                # Check if production matches the start token
                if rhs[0] == startToken:
                    children = []
                    for i in range(len(rhs)):
                        child = TerminalNode(
                            name=rhs[i], parent=self, token=rhs[i])
                        children.append(child)

                    # Link with right sibblings
                    for i in range(len(children) - 1):
                        children[i].next_sibbling = children[i + 1]

                    # Add only the first node of each production to the parent's children,
                    # which will serve as an entrypoint to the subgraph.
                    self.children.append(children[0])

                    mwStartNodes.append(children[0])

            # Add to set of expanded tokens so we can retrieve it if necessary
            self.expandedStartTokens.add(startToken)

        return mwStartNodes


class LanguageModel(object):

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def reset(self):
        raise NotImplementedError()

    def addToken(self, token):
        raise NotImplementedError()

    def getDistributionOverNextTokens(self):
        raise NotImplementedError()

    def nextPossibleTokens(self):
        raise NotImplementedError()


class GrammarLanguageModel(LanguageModel):

    def __init__(self, grammar, useTerminalGroupNodes=False):
        self.grammar = grammar
        self.useTerminalGroupNodes = useTerminalGroupNodes

        vocabulary = Vocabulary.fromGrammar(grammar)
        super(GrammarLanguageModel, self).__init__(vocabulary)

        # NOTE: cache productions in a dictionary for faster look-up
        self.productions = dict()
        for production in self.grammar.productions():
            lhs = production.lhs().symbol()

            if lhs == self.grammar.start().symbol():
                # Add the end marker for the start productions
                rhs = production.rhs() + (Vocabulary.endToken,)
                production = Production(lhs, rhs)

            if lhs in self.productions:
                self.productions[lhs].append(production)
            else:
                self.productions[lhs] = [production]

        # Look for productions with only terminals
        self.terminalgroups = dict()
        if self.useTerminalGroupNodes:
            for productions in six.itervalues(self.productions):
                allTerminals = True
                for production in productions:
                    if not all(not isinstance(p, Nonterminal) for p in production.rhs()):
                        allTerminals = False
                        break

                if allTerminals and len(productions) > 1:
                    tokens = set([production.rhs()[0]
                                  for production in productions])
                    lhs = productions[0].lhs().symbol()
                    self.terminalgroups[lhs] = tokens

        self.reset()

    def reset(self):
        self.pos = 0
        assert (isinstance(self.grammar.start(), Nonterminal))
        label = self.grammar.start().symbol()
        self.start = NonTerminalNode(label)
        self.tree = set([self.start])
        self.frontier = set()
        self._expand_node(self.start)

    def _expand_node(self, node):

        if len(node.children) == 0:
            lhs = node.name
            for production in self.productions[lhs]:
                rhs = production.rhs()

                # Create children and link with parent node
                children = []
                for i in range(len(rhs)):
                    if isinstance(rhs[i], Nonterminal):
                        if self.useTerminalGroupNodes and rhs[i].symbol() in self.terminalgroups:
                            startTokens = self.terminalgroups[rhs[i].symbol()]
                            productions = self.productions[rhs[i].symbol()]
                            child = TerminalGroupNode(name=rhs[i].symbol(
                            ), parent=node, startTokens=startTokens, productions=productions)
                        else:
                            child = NonTerminalNode(
                                name=rhs[i].symbol(), parent=node)
                    else:
                        child = TerminalNode(
                            name=rhs[i], parent=node, token=rhs[i])
                    children.append(child)
                    self.tree.add(child)

                # Link with right sibblings
                for i in range(len(children) - 1):
                    children[i].next_sibbling = children[i + 1]

                # Add only the first node of each production to the parent's children,
                # which will serve as an entrypoint to the subgraph.
                node.children.append(children[0])

                # Add children to tree and continue to expand with left-recursion
                if len(children) > 0:
                    child = children[0]
                    if child.nodeType == NodeType.nonterminal:
                        # Expand child node
                        self._expand_node(child)
                    else:
                        self.frontier.add(child)
        else:
            # Node as already been expanded, so check all the children
            if len(node.children) > 0:
                for child in node.children:
                    if child.nodeType == NodeType.nonterminal:
                        # Expand child node
                        self._expand_node(child)
                    else:
                        self.frontier.add(child)

    def addToken(self, token):

        # Get all matching terminal nodes at the frontier
        nodes = [n for n in self.frontier if n.match(token)]

        # Reset the frontier and update position
        self.frontier = set()

        # Expand terminal groups for given token
        if self.useTerminalGroupNodes:
            newNodes = []
            for node in nodes:
                if node.nodeType == NodeType.terminalgroup:
                    mwStartNodes = node.expandForMultitoken(token)
                    newNodes.extend(mwStartNodes)
                else:
                    newNodes.append(node)
            nodes = newNodes

        # Expand sibblings in the tree starting from nodes on the frontier
        for node in nodes:
            sibbling = node.findNextSibbling()
            if sibbling is not None:
                if sibbling.nodeType == NodeType.nonterminal:
                    self._expand_node(sibbling)
                else:
                    self.frontier.add(sibbling)

        self.pos += 1

    def getDistributionOverNextTokens(self):
        dist = np.zeros((len(self.vocabulary.tokens)), dtype=np.float32)
        tokens = self.nextPossibleTokens()
        for token in tokens:
            dist[self.vocabulary.tokenToIndex(token)] = 1.0
        if len(tokens) > 0:
            dist /= np.sum(dist)
        return dist

    def nextPossibleTokens(self):
        tokens = []
        for n in self.frontier:
            if n.nodeType == NodeType.terminal:
                tokens.append(n.token)
            elif n.nodeType == NodeType.terminalgroup:
                tokens.extend(list(n.startTokens))
        return set(tokens)
