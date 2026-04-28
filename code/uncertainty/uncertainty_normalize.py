"""
Normalize and aggregate the four sentence-level uncertainty signals
(LNLP, MTE, TokenSAR, CLIP-score) and assign confidence levels.

Usage:
    python uncertainty_normalize.py \
        --input_path  ./output/llava_val_mmb.json \
        --output_path ./output/confidence_level.json
"""

import argparse
import copy
import json

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

def z_score_to_sigmoid(mean, std, values):
    """Z-score normalize then apply sigmoid to map values to [0, 1]."""
    values = np.array(values)
    z_scores = (values - mean) / (std if std > 0 else 1e-6)
    normalized_values = 1 / (1 + np.exp(-z_scores))
    return normalized_values


def maxmin_normalized(max_value, min_value, values):
    """Min-max normalize values to [0, 1]."""
    values = np.array(values)
    denom = max_value - min_value
    if denom == 0:
        return np.zeros_like(values)
    normalized_values = (values - min_value) / denom
    return normalized_values


def confidence_level_mapping(result, mean_result_all, std_result_all):
    """Map aggregated uncertainty scores to discrete confidence levels (1-5).

    Level 5 = most confident, level 1 = least confident.
    Boundaries are defined by multiples of std around the mean.
    """
    bins = [
        (0.0, mean_result_all - 1 * std_result_all, 5),
        (mean_result_all - 1 * std_result_all, mean_result_all + 1 * std_result_all, 4),
        (mean_result_all + 1 * std_result_all, mean_result_all + 2 * std_result_all, 3),
        (mean_result_all + 2 * std_result_all, mean_result_all + 3 * std_result_all, 2),
        (mean_result_all + 3 * std_result_all, 1.0, 1),
    ]

    levels = []
    for value in result:
        for lower, upper, level in bins:
            if lower <= value < upper:
                levels.append(level)
                break
    return levels


def correct_length_alignment(data):
    """Ensure all uncertainty signal lists have the same length as sentences.

    Truncates longer lists and pads shorter ones by repeating the last element.
    Returns (corrected_data, error_count).
    """
    error_len = 0
    for item in data:
        llava_result = item.get('llava_reslut')
        if not llava_result:
            continue

        sentences = llava_result.get('sentences', [])
        target_len = len(sentences)

        for key in ('sentence_lnlp', 'sentence_MTE', 'sentence_tokensar', 'sentence_clipscore'):
            lst = llava_result.get(key, [])
            if len(lst) > target_len:
                error_len += 1
                llava_result[key] = lst[:target_len]
            elif len(lst) < target_len:
                error_len += 1
                llava_result[key] = lst + [lst[-1]] * (target_len - len(lst))

    return data, error_len


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize uncertainty signals and assign confidence levels")
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the VLM output JSON (with llava_reslut)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to write the output JSON with confidence levels')
    parser.add_argument('--corrected_path', type=str, default=None,
                        help='Optional path to save length-corrected intermediate data')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        llava_output_data = json.load(f)

    print(f"Loaded {len(llava_output_data)} samples")

    # Step 1: correct length alignment
    corrected_data, error_len = correct_length_alignment(llava_output_data)
    print(f"Length alignment errors corrected: {error_len}")

    if args.corrected_path:
        with open(args.corrected_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, ensure_ascii=False, indent=4)

    # Step 2: collect all signal values across the dataset
    all_lnlp = []
    all_MTE = []
    all_tokensar = []
    all_clipscore = []
    skip_count = 0

    for data_info in tqdm(corrected_data, desc="Collecting signals"):
        llava_result = data_info.get('llava_reslut')
        if not llava_result:
            continue

        s_lnlp = [abs(x) for x in llava_result.get('sentence_lnlp', [])]
        s_MTE = [abs(x) for x in llava_result.get('sentence_MTE', [])]
        s_tokensar = [abs(x) for x in llava_result.get('sentence_tokensar', [])]
        s_clipscore = [abs(x) for x in llava_result.get('sentence_clipscore', [])]

        if len(s_lnlp) == len(s_MTE) == len(s_tokensar) == len(s_clipscore):
            all_lnlp.extend(s_lnlp)
            all_MTE.extend(s_MTE)
            all_tokensar.extend(s_tokensar)
            all_clipscore.extend(s_clipscore)
        else:
            skip_count += 1

    print(f"Signal counts: lnlp={len(all_lnlp)}, MTE={len(all_MTE)}, "
          f"tokensar={len(all_tokensar)}, clipscore={len(all_clipscore)}, "
          f"skipped={skip_count}")

    if not all_lnlp:
        print("ERROR: No valid signal data found. Exiting.")
        return

    # Print statistics
    for name, vals in [('lnlp', all_lnlp), ('MTE', all_MTE),
                       ('tokensar', all_tokensar), ('clipscore', all_clipscore)]:
        print(f"  {name}: mean={np.mean(vals):.6f}, std={np.std(vals):.6f}, "
              f"max={max(vals):.6f}, min={min(vals):.6f}")

    # Step 3: normalize each signal with min-max and aggregate
    normalized_data = []
    result_all = []

    for data_info in tqdm(corrected_data, desc="Normalizing"):
        new_data_info = copy.deepcopy(data_info)
        llava_result = data_info.get('llava_reslut')
        if not llava_result:
            continue

        sentence_lnlp = llava_result.get('sentence_lnlp', [])
        sentence_MTE = llava_result.get('sentence_MTE', [])
        sentence_tokensar = llava_result.get('sentence_tokensar', [])
        sentence_clipscore = llava_result.get('sentence_clipscore', [])

        new_lnlp = maxmin_normalized(max(all_lnlp), min(all_lnlp), sentence_lnlp)
        new_MTE = maxmin_normalized(max(all_MTE), min(all_MTE), sentence_MTE)
        new_tokensar = maxmin_normalized(max(all_tokensar), min(all_tokensar), sentence_tokensar)
        new_clipscore = maxmin_normalized(max(all_clipscore), min(all_clipscore), sentence_clipscore)

        if len(new_lnlp) == len(new_MTE) == len(new_tokensar) == len(new_clipscore):
            # Average of 4 signals (LNLP + MTE + TokenSAR + CLIPscore)
            result = [float((x + y + z + u) / 4)
                      for x, y, z, u in zip(new_lnlp, new_MTE, new_tokensar, new_clipscore)]
            result_all.extend(result)

            mean_result_all = np.mean(result_all)
            std_result_all = np.std(result_all)

            new_data_info['sentence_confidence'] = result
            confidence_level = confidence_level_mapping(result, mean_result_all, std_result_all)
            new_data_info['sentence_confidence_level'] = confidence_level
            normalized_data.append(new_data_info)
        else:
            print(f"ERROR: length mismatch for image_id={data_info.get('image_id')}")

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)

    print(f"Wrote {len(normalized_data)} samples to {args.output_path}")


if __name__ == '__main__':
    main()
