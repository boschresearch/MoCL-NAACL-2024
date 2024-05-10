""" Utility classes and functions related to MoCL (NAACL 2024).
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import string
from rouge import rouge_scorer


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        gold = [gold]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics