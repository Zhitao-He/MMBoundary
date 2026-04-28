"""
MECE and ECE calibration metrics for MMBoundary evaluation.

- MECE (Mean Expected Calibration Error): Per-claim calibration error.
  For each predicted claim, compares its confidence (normalized to [0,1]) against
  a binary correctness indicator (1 if matched to a reference, 0 otherwise).
  Returns the mean absolute difference across all claims.

- ECE (Expected Calibration Error): Overall calibration error.
  Compares the mean normalized confidence across all claims against the
  overall accuracy score, returning the absolute difference.

Confidence values are integers in [0, 5] and normalized by dividing by 5.
"""


def calculate_ece(pred_list, pred_conf_value, acc_pred):
    """
    Calculate the Expected Calibration Error (ECE) focusing only on the last prediction.

    Parameters:
    - pred_list (list): List of predicted sentence indices.
    - pred_conf_value (list): Confidence values for each prediction (e.g., discrete values like 0-5).
    - match_pred (list): Indices of correctly matched predictions (matching with the reference).

    Returns:
    - ECE (float): The calculated ECE value for the last prediction.
    """
    # Maximum confidence value for normalization
    max_value = 5

    conf_value_list = [conf / max_value for conf in pred_conf_value]
    conf_value = sum(conf_value_list) / len(conf_value_list)

    # Calculate the absolute difference for the last prediction
    absolute_difference_last = abs(acc_pred - conf_value)

    # Return the calculated ECE (which is just the absolute difference in this case)
    return absolute_difference_last


def calculate_mece(pred_list, pred_conf_value, match_pred):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    - pred_list (list): List of predicted sentence indices.
    - pred_conf_value (list): Confidence values for each prediction (e.g., discrete values like 0-5).
    - match_pred (list): Indices of correctly matched predictions (matching with the reference).

    Returns:
    - ECE (float): The calculated ECE value.
    """
    # Maximum confidence value for normalization
    max_value = 5

    # 1. Normalize confidence values to the range [0, 1]
    conf_values = [conf / max_value for conf in pred_conf_value]

    # 2. Calculate the indicator function (1 for correct, 0 for incorrect)
    # Check if each prediction index is in match_pred
    indicator = [1 if idx in match_pred else 0 for idx in range(len(pred_list))]

    # 3. Compute the absolute differences and calculate ECE
    absolute_differences = [abs(indicator[i] - conf_values[i]) for i in range(len(pred_list))]
    if len(pred_list) > 0:
        ECE = sum(absolute_differences) / len(pred_list)
    else:
        ECE = 0

    # Return the calculated ECE
    return ECE


def MECE_compute(predicted_sentences, predicted_values, sentence_result):
    """Compute per-claim MECE from predicted sentences, confidence values, and match results."""
    match_pred = sentence_result.get('match_pred', [])
    MECE_score = calculate_mece(predicted_sentences, predicted_values, match_pred)
    return MECE_score


def ECE_compute(predicted_sentences, predicted_values, sentence_result):
    """Compute overall ECE from predicted sentences, confidence values, and accuracy."""
    acc_pred = sentence_result.get('acc', [])
    ECE_score = calculate_ece(predicted_sentences, predicted_values, acc_pred)
    return ECE_score
