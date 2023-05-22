from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    def iterate_alignment(alignment):
        for pair in alignment.sure:
            yield pair
        for pair in alignment.possible:
            yield pair

    intersection = 0
    total_predicted = 0
    for alignment, predicted_alignment in zip(reference, predicted):
        total_predicted += len(predicted_alignment)
        intersection += len(set(iterate_alignment(alignment)).intersection(predicted_alignment))

    return intersection, total_predicted


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    def iterate_sure_alignment(alignment):
        for pair in alignment.sure:
            yield pair

    intersection = 0
    total_predicted = 0
    for alignment, predicted_alignment in zip(reference, predicted):
        total_predicted += len(alignment.sure)
        intersection += len(set(iterate_sure_alignment(alignment)).intersection(predicted_alignment))

    return intersection, total_predicted


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    intersection_A_P, total_predicted = compute_precision(reference, predicted)
    intersection_A_S, total_predicted_S = compute_recall(reference, predicted)

    return 1.0 - (intersection_A_P + intersection_A_S) / (total_predicted + total_predicted_S)
