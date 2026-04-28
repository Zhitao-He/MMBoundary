"""
Inverse mapping of confidence statements to discrete confidence levels.

Given a set of reference confidence statements (organized by confidence level)
and a batch of input sentences, this module classifies each sentence into the
best-matching confidence level using cosine similarity of sentence embeddings.

Requires:
  - sentence-transformers
  - A SentenceTransformer model (e.g., all-MiniLM-L6-v2)
  - A JSON file of confidence statements with structure:
    [{"confidence_level": int, "confidence_name": str, "statement": [str, ...]}, ...]
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import json


def prepare_statements_embeddings(statements, model):
    """
    Encode all confidence statements into embeddings for later use.

    Parameters:
    - statements (list): A list of confidence levels, each containing statements and their labels.
    - model (SentenceTransformer): The model to compute embeddings.

    Returns:
    - dict: A dictionary mapping confidence levels to their precomputed embeddings.
    """
    embeddings = {}
    for level in statements:
        confidence_level = level["confidence_level"]
        confidence_name = level["confidence_name"]
        level_statements = level["statement"]
        # Encode statements for this confidence level
        embeddings[confidence_level] = {
            "confidence_name": confidence_name,
            "embeddings": model.encode(level_statements)
        }
    return embeddings


def classify_batch(sentences, statements_embeddings, model):
    """
    Classify a batch of sentences into confidence levels.

    Parameters:
    - sentences (list): A batch of sentences to classify.
    - statements_embeddings (dict): Precomputed embeddings of confidence statements.
    - model (SentenceTransformer): The model to compute embeddings for input sentences.

    Returns:
    - list: A list of dictionaries containing the best matching confidence level for each sentence.
    """
    # Encode the input sentences
    sentence_embeddings = model.encode(sentences)
    results = []
    values = []

    for sentence_idx, sentence_embedding in enumerate(sentence_embeddings):
        best_score = -float('inf')
        best_level = None

        # Compare with each confidence level's embeddings
        for confidence_level, data in statements_embeddings.items():
            level_embeddings = data["embeddings"]
            confidence_name = data["confidence_name"]

            # Compute cosine similarity with all embeddings in this level
            scores = util.cos_sim(sentence_embedding, level_embeddings).squeeze().numpy()
            total_score = np.sum(scores) / len(scores)

            # Update the best level if necessary
            if total_score > best_score:
                best_score = total_score
                best_level = {
                    "confidence_level": confidence_level,
                    "confidence_name": confidence_name,
                    "score": total_score
                }
        values.append(best_level["confidence_level"])

        # Store result for this sentence
        results.append({
            "sentence": sentences[sentence_idx],
            "confidence_level": best_level["confidence_level"],
            "confidence_name": best_level["confidence_name"]
        })

    return results, values


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify sentences into confidence levels via inverse mapping."
    )
    parser.add_argument(
        "--statement_path", type=str, required=True,
        help="Path to the confidence_statement.json file."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to a SentenceTransformer model (e.g., all-MiniLM-L6-v2)."
    )
    args = parser.parse_args()

    with open(args.statement_path, 'r', encoding='utf-8') as file:
        statements = json.load(file)

    # Load the model
    model = SentenceTransformer(args.model_path)

    # Precompute embeddings for statements
    statements_embeddings = prepare_statements_embeddings(statements, model)

    # Example batch of sentences to classify
    sentences = [
        "and I'm absolutely sure about this.",
        "with no doubts at all.",
        "and this is completely certain.",
        "and I'm fully confident in this.",
        "with total certainty.",
        "and this answer is definitive.",
        "with absolute clarity.",
        "and this is undoubtedly correct.",
        "and I'm sure this is accurate.",
        "with full assurance.",
        "and this is entirely reliable.",
        "and I know this is correct.",
        "with zero hesitation.",
        "and this holds with certainty.",
        "and I stand by this completely.",
        "and it's unquestionably right.",
        "and this is fully accurate.",
        "with complete confidence.",
        "and this is beyond any doubt.",
        "and I'm certain this is true.",
        "and this is fully verified.",
        "with full trust in this.",
        "and I guarantee this is right.",
        "and it's perfectly accurate.",
        "with nothing to question.",
        "and this is indisputable.",
        "and I have no reservations here.",
        "and it's entirely dependable.",
        "with total conviction.",
        "and this is rock-solid.",
        "and it's absolutely flawless.",
        "and this answer is airtight.",
        "and this conclusion is final.",
        "with no uncertainties remaining.",
        "and it's fully supported.",
        "and this is entirely certain.",
        "and this is perfectly correct.",
        "and I'm 100% confident in this."
    ]

    # Classify the batch
    results, values = classify_batch(sentences, statements_embeddings, model)
    for idx, result in enumerate(results):
        print(f"{idx} Predicted confidence level: {result['confidence_level']} ({result['confidence_name']})")

    print('confidence values:', values)
