import numpy as np
import itertools
from sentence_transformers import CrossEncoder

# Default model for token similarity computation.
# Override via the model_name parameter in calculate_token_similarity().
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/stsb-roberta-large"


def extract_by_sentences(data):
    sentence_list = []
    tokens_list = []
    token_logprob_list = []

    current_sentence = ""
    current_tokens = []
    current_logprobs = []

    sentence_end_markers = {'.', '!', '?'}

    for token_group in data:
        token_info = token_group[0]
        token_text = token_info['text']
        token_logprob = token_info['logprob']

        current_sentence += token_text
        current_tokens.append(token_text)
        current_logprobs.append(token_logprob)

        if token_text.strip() and token_text.strip()[-1] in sentence_end_markers:
            sentence_list.append(current_sentence.strip())
            tokens_list.append(current_tokens)
            token_logprob_list.append(current_logprobs)

            current_sentence = ""
            current_tokens = []
            current_logprobs = []

    if current_sentence:
        sentence_list.append(current_sentence.strip())
        tokens_list.append(current_tokens)
        token_logprob_list.append(current_logprobs)

    return sentence_list, tokens_list, token_logprob_list

def calculate_token_similarity(sentence: str, tokens: list, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> np.ndarray:
    """
    Calculate the token similarity for a given sentence and its tokens using a cross-encoder model.

    Args:
        sentence: The input sentence.
        tokens: List of tokens to compare similarity.
        model_name: The name or path of the cross-encoder model.
                    Defaults to DEFAULT_CROSS_ENCODER_MODEL.

    Returns:
        A similarity score list for the tokens.
    """
    cross_encoder = CrossEncoder(model_name)

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    token_scores = []

    if len(tokens) > 1:
        cropped_tokens = list(itertools.combinations(tokens, len(tokens) - 1))[::-1]
        raw_text = " " + " ".join(tokens)

        batches = [
            (
                raw_text,
                " ".join(list(t)),
            )
            for t in cropped_tokens
        ]

        token_scores = cross_encoder.predict(batches)

        is_special_tokens = np.isin(tokens, special_tokens)
        token_scores[is_special_tokens] = 1
    else:
        token_scores = np.array([0.5] * len(tokens))

    return list(token_scores)

def calculate_token_sar(tokens, token_logprob, token_similarity):
    """
    Calculate the TokenSAR for a given sequence.

    Args:
        tokens: List of tokens in the sequence.
        token_logprob: Log probabilities of each token.
        token_similarity: Similarity scores of each token.

    Returns:
        The calculated TokenSAR value for the sequence.
    """
    log_likelihoods = -np.array(token_logprob)
    R_t = 1 - np.array(token_similarity)

    R_t_norm = R_t / R_t.sum()

    E_t = log_likelihoods * R_t_norm

    token_sar = E_t.sum()

    return token_sar
