"""
Process-level similarity scoring using CrossEncoder.

Computes an unordered matching between predicted claims and reference claims:
  - For each prediction, checks if any reference exceeds the similarity threshold.
  - Tracks precision (how many predictions matched) and recall (how many references
    were covered), along with matched indices for downstream calibration metrics.

Requires:
  - sentence-transformers (CrossEncoder)
  - A CrossEncoder model passed as a parameter (e.g., cross-encoder/stsb-roberta-base).
"""

from collections import defaultdict
from typing import List
from sentence_transformers import CrossEncoder


def split_into_entries(text):
    """
    Helper function to split a list of sentences.
    In this context, it simply returns the input as-is, assuming input is a list of strings.
    """
    return text


def compute_score(predictions, references, model):
    """
    Compute similarity scores between predictions and references using CrossEncoder.

    Args:
        predictions (List[str]): List of predicted sentences.
        references (List[str]): List of reference sentences.
        model: Preloaded CrossEncoder model.

    Returns:
        List[List[float]]: A list of lists containing similarity scores.
    """
    scores = []
    for pred in predictions:
        pairs = [(pred, ref) for ref in references]
        pred_scores = model.predict(pairs)
        scores.append(pred_scores.tolist())
    return scores


def unordered_row_score(
    preds: List[str],
    refs: List[str],
    model,
    match_threshold: float
):
    """
    Compute unordered row-level score between predictions and references.

    Args:
        preds (List[str]): List of predicted sentences.
        refs (List[str]): List of reference sentences.
        model: Preloaded CrossEncoder model.
        match_threshold (float): Threshold for considering a prediction matches a reference.

    Returns:
        dict: A dictionary containing precise, positive_predictions, covered, and relevant metrics.
               Also includes matched indices between predictions and references.
    """
    output = {
        'precise': 0,
        'positive_predictions': len(preds),
        'covered': set(),
        'relevant': len(refs),
        'match_pred': [],   # Matched prediction indices
        'match_refer': []   # Matched reference indices
    }

    # Compute similarity scores
    similarity_scores = compute_score(
        predictions=preds,
        references=refs,
        model=model
    )

    # For each predicted sentence, calculate scores against all reference sentences
    for pred_idx, scores in enumerate(similarity_scores):
        if max(scores) > match_threshold:
            output['precise'] += 1

        # Add all reference indices covered by this prediction
        for ref_idx, score in enumerate(scores):
            if score > match_threshold:
                output['covered'].add(ref_idx)
                output['match_pred'].append(pred_idx)
                output['match_refer'].append(ref_idx)

    output['covered'] = len(output['covered'])
    return output


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute process-level similarity between predicted and reference claims."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to a CrossEncoder model (e.g., cross-encoder/stsb-roberta-base)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Similarity threshold for matching (default: 0.5)."
    )
    args = parser.parse_args()

    reference_sentences = [
        "The image shows a man standing by the roadside with bags",
        "He appears to be looking towards the street",
        "The man has a suitcase and a smaller bag beside him",
        "People typically wait for transportation while standing with luggage",
        "The common types of transportation in such settings include cabs",
        "So, the answer is cab"
    ]

    predicted_sentences = [
        "The man is standing by the bags, which include a suitcase and a handbag, on the side of the road",
        "He is likely waiting for transportation, such as a taxi or a ride from a friend or family member",
        "The presence of multiple cars in the background suggests that he is in a busy area, "
        "which could be a reason for him to seek transportation quickly"
    ]

    model = CrossEncoder(args.model_path)

    result = unordered_row_score(
        preds=predicted_sentences,
        refs=reference_sentences,
        model=model,
        match_threshold=args.threshold
    )
    print("Result:", result)
