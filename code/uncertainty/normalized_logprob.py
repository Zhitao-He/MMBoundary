import math


def get_token_logprob(raw_data):
    result = []
    for item in raw_data:
        formatted_item = [{"token": token, "text": value.decoded_token, "logprob": value.logprob} for token, value in item.items()]
        result.append(formatted_item)

    return result

def calculate_sentences_length_normalized_logprob(data):
    sentence_logprobs = []
    sentence_texts = []
    total_logprob = 0.0
    total_length = 0
    sentence = []
    sentence_text = []

    for token_group in data:
        best_candidate = token_group[0]
        logprob = best_candidate['logprob']
        text = best_candidate['text']

        if text:
            sentence.append(logprob)
            sentence_text.append(text)
            total_logprob += logprob
            total_length += 1

        if text in ['.', '!', '?']:
            if total_length > 0:
                sentence_logprob = -total_logprob / total_length
            else:
                sentence_logprob = float('inf')
            sentence_logprobs.append(sentence_logprob)
            sentence_texts.append(''.join(sentence_text))

            sentence = []
            sentence_text = []
            total_logprob = 0.0
            total_length = 0

    if total_length > 0:
        sentence_logprob = -total_logprob / total_length
        sentence_logprobs.append(sentence_logprob)
        sentence_texts.append(''.join(sentence_text))

    return sentence_texts, sentence_logprobs
