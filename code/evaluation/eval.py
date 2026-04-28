"""
MMBoundary Evaluation Pipeline (eval.py)

Main entry point for evaluating VLM responses on the MMBoundary benchmark.
Computes the following metrics:

  Accuracy          -- GPT-4o judges whether the model's answer contains the gold
                       answer (binary 0/1, averaged across samples).

  Process_precision -- Fraction of predicted claims that match at least one
                       reference claim (CrossEncoder similarity > threshold).

  Process_recall    -- Fraction of reference claims covered by at least one
                       predicted claim.

  Process_F1        -- Harmonic mean of Process_precision and Process_recall.

  ECE               -- Expected Calibration Error. Compares the mean normalized
                       confidence of all claims against overall accuracy, measuring
                       how well the model's aggregate confidence tracks correctness.

  MECE              -- Mean Expected Calibration Error. Per-claim calibration:
                       for each claim, compares its confidence to a binary
                       correctness indicator (matched vs. unmatched), then averages.

Usage:
  python eval.py \\
    --dataset_name aokvqa \\
    --response_path /path/to/responses.json \\
    --cross_encoder_model /path/to/roberta \\
    --statement_model /path/to/all_MiniLM_L6-v2 \\
    --confidence_statement_path /path/to/confidence_statement.json

Environment:
  OPENAI_API_KEY must be set (used by the Accuracy metric via GPT-4o).
"""

import os
import json
import re
import ast
import logging
import argparse
from tqdm import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
from openai import OpenAI

from inverse_mapping import prepare_statements_embeddings, classify_batch
from process_score_similarity import unordered_row_score
from MECE import MECE_compute, ECE_compute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

prompt_accuracy_aokvqa = """
    You will be given a question, its corresponding gold answer, and a candidate answer. The gold answer might be a single word or a list of words (the words in the list are generally synonyms). The candidate answer may contain multiple sentences.
    Your task is to determine whether the candidate answer contains the gold answer (it only needs to include one word or a similar word to the gold answer).
    If the candidate answer includes the gold answer, return 1 (Correct Answer) without any explanation. If the candidate answer does not include the gold answer, return 0 (Incorrect Answer) without any explanation.
    Please refer to the following examples:

    Example 1
    Question: What is the man by the bags awaiting?
    Gold answer: ["ride", "bus", "taxi", "travelling", "traffic", "cab", "his ride"]
    Candidate answer: The man is standing by the bags, which include a suitcase and a handbag, on the side of the road, [and it aligns logically.] He is likely waiting for transportation, such as a taxi or a ride from a friend or family member, [and it's perfectly accurate.] The presence of multiple cars in the background suggests that he is in a busy area, which could be a reason for him to seek transportation quickly, [and this appears very reliable.]
    Result: 1

    Example 2
    Question: What is the name of the painting on the outside of the blue train?
    Gold answer: ["graffiti", "blue", "railway", "jeans"]
    Candidate answer: The image shows a blue train on the tracks, with a train station in the background, [though doubts still remain.] There is no visible painting on the outside of the train, [and this feels reliable to me.] The train appears to be a standard passenger train, and there are no distinctive markings or logos that would indicate a specific painting or artwork, [and this appears very reliable.] The focus of the image is on the train and the tracks, with no additional details that would allow for a definitive answer to the question, [though it's not beyond scrutiny.]
    Result: 0

    Example 3
    Question: What mood do the cows seem to be in?
    Gold answer: ["good", "happy", "curious"]
    Candidate answer: The cows in the image appear to be in a calm and relaxed mood, [and this is well-reasoned.] They are grazing peacefully in the grassy field, which suggests that they are comfortable and undisturbed, [but nothing is infallible.] The presence of a man sitting nearby also indicates that the cows are not bothered by human presence, further supporting the idea that they are at ease in their environment, [and it aligns logically.]
    Result: 1

    Now, please answer:
    Gold answer: {gold_answer}
    Candidate answer: {candidate_answer}
    Result:
"""

prompt_decompose2sents = """
    You will receive a piece of content containing multiple claims and corresponding confidence statements. Your task is to split the content into multiple sentences (based on periods). Then, for each sentence, separate the claim from its corresponding confidence statement. Do not modify any text; simply divide the content as instructed.

    Please format your output in JSON, as follows:
    Output: {{
            'Claim': [],
            'Confidence': []
            }}

    Refer to the following example:
    Content: In the image, besides skateboarding, people can also enjoy watching others skateboard, [and this seems highly dependable.]  There are several people standing around the skate park, observing the skateboarders performing tricks, [but there's a slim chance of error.]  This suggests that the skate park is not only a place for skateboarders to practice and showcase their skills but also a social gathering spot for people to watch, learn, and appreciate the sport, [and I feel strongly about this.]
    Output: {{
            'Claim': ['In the image, besides skateboarding, people can also enjoy watching others skateboard', 'There are several people standing around the skate park, observing the skateboarders performing tricks', 'This suggests that the skate park is not only a place for skateboarders to practice and showcase their skills but also a social gathering spot for people to watch, learn, and appreciate the sport'],
            'Confidence': ['and this seems highly dependable.', 'but there\\'s a slim chance of error.', 'and I feel strongly about this.']
            }}

    Content: The silver vehicle by the building is a train, [and I believe this is right.]  It is parked at a train station, and there are people walking around the platform, [and I'm absolutely sure about this.]  The presence of the train and people walking around the platform indicates that it is a train station, and the train is used for transportation, [and this matches the evidence.]
    Output: {{
            'Claim': ['The silver vehicle by the building is a train', 'It is parked at a train station, and there are people walking around the platform', 'The presence of the train and people walking around the platform indicates that it is a train station, and the train is used for transportation'],
            'Confidence': ['and I believe this is right.', 'and I\\'m absolutely sure about this.', 'and this matches the evidence.']
            }}

    Now, please answer:
    Content: {content}
    Output: {{
            'Claim': [],
            'Confidence': []
            }}

    Your output:
        Output:
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def call_gpt4o(text_input):
    """Call GPT-4o via the OpenAI API. Reads OPENAI_API_KEY from the environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )
    client = OpenAI(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text_input},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT-4o call failed: {e}")
        return None


def extract_and_convert(text_output):
    """Extract a Python dict from GPT output text and safely parse it."""
    match = re.search(r'{(.*)}', text_output, re.DOTALL)
    if match:
        dict_content = match.group(0).strip()

        # Handle quote replacement to avoid conflicts
        temp_content = re.sub(r'"', '$$', dict_content)
        formatted_text = (
            temp_content
            .replace("\\'s", '##s')
            .replace("\\'", "@@")
            .replace("'", '"')
        )
        formatted_text = (
            formatted_text
            .replace('$$', "'")
            .replace("##s", "\\'s")
            .replace("@@", "\\'")
        )

        try:
            result_dict = ast.literal_eval(formatted_text)
            return result_dict
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error converting text to dictionary: {e}")
            return None
    else:
        logger.warning("No dictionary found in the text.")
        return None


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_accuracy(dataset_name, response, reference_answer, max_retries=3):
    """
    Use GPT-4o to judge whether *response* contains the gold *reference_answer*.

    Returns 1 (correct), 0 (incorrect), or None on failure.
    """
    if dataset_name == 'aokvqa':
        prompt = prompt_accuracy_aokvqa.format(
            gold_answer=reference_answer, candidate_answer=response
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for attempt in range(max_retries):
        try:
            output = call_gpt4o(prompt)
            if isinstance(output, (int, str)) and int(output) in [0, 1]:
                return int(output)
            else:
                logger.warning(
                    f"Invalid output: {output} on attempt {attempt + 1}/{max_retries}."
                )
        except Exception as e:
            logger.error(
                f"GPT-4o call failed on attempt {attempt + 1}/{max_retries}: {e}"
            )

    logger.error("Max retries reached. Returning None.")
    return None


def compute_process_score(
    response,
    reference_answer,
    statements_embeddings,
    statement_model,
    sentence_score_model,
    match_threshold=0.5,
    max_retries=3,
):
    """
    Decompose the response into claims + confidence statements via GPT-4o,
    then compute process-level precision/recall and confidence values.

    Returns:
        (sentence_result, predicted_sentences, predicted_values)
    """
    prompt = prompt_decompose2sents.format(content=response)

    for attempt in range(max_retries):
        try:
            gpt_output = call_gpt4o(prompt)
            if gpt_output:
                parsed = extract_and_convert(gpt_output)
                if parsed is None:
                    logger.warning(f"Parsing failed on attempt {attempt + 1}.")
                    continue

                predicted_sentences = parsed['Claim']
                predicted_statements = parsed['Confidence']

                # Inverse-map confidence statements to discrete levels
                results, predicted_values = classify_batch(
                    predicted_statements, statements_embeddings, statement_model
                )

                # Compute sentence-level similarity matching
                sentence_result = unordered_row_score(
                    preds=predicted_sentences,
                    refs=reference_answer,
                    model=sentence_score_model,
                    match_threshold=match_threshold,
                )
                return sentence_result, predicted_sentences, predicted_values
        except Exception as e:
            logger.error(
                f"Process_score failed on attempt {attempt + 1}/{max_retries}: {e}"
            )

    logger.error("Max retries reached for Process_score. Returning empty results.")
    return {}, [], []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MMBoundary evaluation pipeline. Computes Accuracy, Process F1, ECE, and MECE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="aokvqa",
        help="Dataset name for accuracy prompt selection (default: aokvqa).",
    )
    parser.add_argument(
        "--response_path",
        type=str,
        required=True,
        help="Path to the JSON file containing VLM responses.",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Optional separate JSON file with reference answers. "
             "If not provided, references are read from the response file "
             "(fields: 'keys_filtered.Image_level' + 'keys_filtered.Reasoning_level').",
    )
    parser.add_argument(
        "--cross_encoder_model",
        type=str,
        required=True,
        help="Path or name of the CrossEncoder model for process similarity "
             "(e.g., cross-encoder/stsb-roberta-base).",
    )
    parser.add_argument(
        "--statement_model",
        type=str,
        required=True,
        help="Path or name of the SentenceTransformer model for confidence mapping "
             "(e.g., sentence-transformers/all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--confidence_statement_path",
        type=str,
        required=True,
        help="Path to the confidence_statement.json file.",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for claim matching (default: 0.5).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    logger.info(f"Loading CrossEncoder model from: {args.cross_encoder_model}")
    sentence_score_model = CrossEncoder(args.cross_encoder_model)

    logger.info(f"Loading SentenceTransformer model from: {args.statement_model}")
    stmt_model = SentenceTransformer(args.statement_model)

    # ------------------------------------------------------------------
    # Load confidence statements and precompute embeddings
    # ------------------------------------------------------------------
    logger.info(f"Loading confidence statements from: {args.confidence_statement_path}")
    with open(args.confidence_statement_path, 'r', encoding='utf-8') as f:
        statements = json.load(f)
    statements_embeddings = prepare_statements_embeddings(statements, stmt_model)

    # ------------------------------------------------------------------
    # Load response data
    # ------------------------------------------------------------------
    logger.info(f"Loading response data from: {args.response_path}")
    with open(args.response_path, 'r', encoding='utf-8') as f:
        response_data = json.load(f)

    # Optionally load separate reference file
    reference_data = None
    if args.reference_path:
        logger.info(f"Loading reference data from: {args.reference_path}")
        with open(args.reference_path, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    accuracy_list = []
    process_precise_list = []
    process_pos_pred_list = []
    process_covered_list = []
    process_relevant_list = []
    ece_score_list = []
    mece_score_list = []

    for idx, data in enumerate(tqdm(response_data, desc="Evaluating")):
        predicted_answer = data['predicted_answer']

        # Resolve reference answer
        if reference_data is not None:
            ref_entry = reference_data[idx]
        else:
            ref_entry = data

        ref_level = ref_entry.get("keys_filtered", {})
        reference_answer = (
            ref_level.get("Image_level", [])
            + ref_level.get("Reasoning_level", [])
        )

        # --- Accuracy ---
        acc_result = compute_accuracy(
            args.dataset_name, predicted_answer, reference_answer
        )
        if acc_result is not None:
            accuracy_list.append(acc_result)

        # --- Process score ---
        sentence_result, predicted_sentences, predicted_values = compute_process_score(
            predicted_answer,
            reference_answer,
            statements_embeddings,
            stmt_model,
            sentence_score_model,
            match_threshold=args.match_threshold,
        )

        process_precise_list.append(sentence_result.get('precise', 0))
        process_pos_pred_list.append(sentence_result.get('positive_predictions', 0))
        process_covered_list.append(sentence_result.get('covered', 0))
        process_relevant_list.append(sentence_result.get('relevant', 0))

        # --- Calibration ---
        if predicted_sentences and predicted_values:
            mece = MECE_compute(predicted_sentences, predicted_values, sentence_result)
            mece_score_list.append(mece)

            if acc_result is not None:
                # Inject accuracy into sentence_result for ECE computation
                sentence_result_with_acc = dict(sentence_result)
                sentence_result_with_acc['acc'] = acc_result
                ece = ECE_compute(
                    predicted_sentences, predicted_values, sentence_result_with_acc
                )
                ece_score_list.append(ece)

    # ------------------------------------------------------------------
    # Aggregate and print results
    # ------------------------------------------------------------------
    accuracy_num = (
        sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
    )
    precision_num = (
        sum(process_precise_list) / sum(process_pos_pred_list)
        if sum(process_pos_pred_list) > 0 else 0.0
    )
    recall_num = (
        sum(process_covered_list) / sum(process_relevant_list)
        if sum(process_relevant_list) > 0 else 0.0
    )
    f1_num = (
        2 * precision_num * recall_num / (precision_num + recall_num)
        if (precision_num + recall_num) > 0 else 0.0
    )
    ece_num = (
        sum(ece_score_list) / len(ece_score_list) if ece_score_list else 0.0
    )
    mece_num = (
        sum(mece_score_list) / len(mece_score_list) if mece_score_list else 0.0
    )

    print("\n" + "=" * 60)
    print("MMBoundary Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy:          {accuracy_num:.4f}")
    print(f"  Process_precision: {precision_num:.4f}")
    print(f"  Process_recall:    {recall_num:.4f}")
    print(f"  Process_F1:        {f1_num:.4f}")
    print(f"  ECE:               {ece_num:.4f}")
    print(f"  MECE:              {mece_num:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
