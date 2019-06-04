# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for calculating loss, accuracy, and other model metrics.

Metrics:
 - Padded loss, accuracy, and negative log perplexity. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py
 - BLEU approximation. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
 - ROUGE score. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""

import collections
import math
from collections import Counter
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]

    if average:
        return sum(scores) / len(scores)

    return scores


def f1_risk(predictions, targets):
    scores = f1_score(predictions, targets, average=False)
    return [1 - s for s in scores]


def bleu_score(predictions, targets): #todo this is redundant: remove it
    scores = [1 - compute_bleu([prediction], [target]) for prediction, target in zip(predictions, targets)]
    #bleu = compute_bleu(labels, predictions)
    return scores


def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)


def _get_ngrams(n, text):
    """Calculates n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
    """Computes ROUGE-N f1 score of two text collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Args:
    eval_sentences: Predicted sentences.
    ref_sentences: Sentences from the reference set
    n: Size of ngram.  Defaults to 2.

  Returns:
    f1 score for ROUGE-N
  """
    f1_scores = []
    for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
        eval_ngrams = _get_ngrams(n, eval_sentence)
        ref_ngrams = _get_ngrams(n, ref_sentence)
        ref_count = len(ref_ngrams)
        eval_count = len(eval_ngrams)

        # Count the overlapping ngrams between evaluated and reference
        overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
        overlapping_count = len(overlapping_ngrams)

        # Handle edge case. This isn't mathematically correct, but it's good enough
        if eval_count == 0:
            precision = 0.0
        else:
            precision = float(overlapping_count) / eval_count
        if ref_count == 0:
            recall = 0.0
        else:
            recall = float(overlapping_count) / ref_count
        f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

    # return overlapping_count / reference_count
    return np.mean(f1_scores, dtype=np.float32)


def rouge_l_sentence_level(eval_sentences, ref_sentences):
    """Computes ROUGE-L (sentence level) of two collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    eval_sentences: The sentences that have been picked by the summarizer
    ref_sentences: The sentences from the reference set

  Returns:
    A float: F_lcs
  """

    f1_scores = []
    for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
        m = float(len(ref_sentence))
        n = float(len(eval_sentence))
        lcs = _len_lcs(eval_sentence, ref_sentence)
        f1_scores.append(_f_lcs(lcs, m, n))
    return np.mean(f1_scores, dtype=np.float32)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

  The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _f_lcs(llcs, m, n):
    """Computes the LCS-based F-measure score.

  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs


if __name__ == "__main__":
    reference_corpus = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    translation_corpus = [['this', 'is', 'a', 'small', 'test']]

    bleu = compute_bleu(reference_corpus, translation_corpus)
    print(bleu)


    # two references for one document
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    smooth_func = SmoothingFunction().method3 #NIST geometric sequence smoothing
    references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
    candidates = [['this', 'is', 'a', 'small', 'test']]
    score = corpus_bleu(references, candidates, smoothing_function=smooth_func)
    print(score)