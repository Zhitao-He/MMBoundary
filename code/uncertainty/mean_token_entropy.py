import math


def calculate_sentence_entropy(data):
    sentence_texts = []
    sentence_entropies = []

    current_sentence = []
    current_sentence_text = []

    for token_data in data:
        selected_token = token_data[0]
        logprob = selected_token['logprob']
        text = selected_token['text']

        if text in ['.', '?', '!']:
            if current_sentence:
                sentence_texts.append(' '.join(current_sentence_text))
                sentence_entropies.append(calculate_mean_token_entropy(current_sentence))
                current_sentence = []
                current_sentence_text = []
        else:
            current_sentence.append(token_data)
            current_sentence_text.append(text)

    if current_sentence:
        sentence_texts.append(' '.join(current_sentence_text))
        sentence_entropies.append(calculate_mean_token_entropy(current_sentence))

    return sentence_texts, sentence_entropies

def calculate_mean_token_entropy(data):
    total_entropy = 0
    L = len(data)

    for token_data in data:
        logprobs = [item['logprob'] for item in token_data if item['logprob'] != -float('inf')]

        if not logprobs:
            continue

        probs = [math.exp(logprob) for logprob in logprobs]
        total_prob = sum(probs)
        probs = [prob / total_prob for prob in probs]

        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        total_entropy += entropy

    mean_token_entropy = total_entropy / L if L > 0 else 0
    return mean_token_entropy
