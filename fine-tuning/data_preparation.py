import pandas as pd
import json

# Paths
# input_csv = '../data/raw_sample_data.csv'
# input_csv = '../data/raw_sample_data_2.csv'
input_csv = '../data/raw_sample_data_3.csv'
# input_csv = '../data/raw_sample_data_4.csv'
# input_csv = '../data/2024-09-25-nikki-qa.csv'
# input_csv = '../data/mismatches.csv'
output_jsonl_train = '../outputs/tmp_sample_finetune_training.jsonl'
output_jsonl_val = '../outputs/tmp_sample_finetune_validation.jsonl'

# Split ratio for train and validation
train_ratio = 0.8


def load_data(input_csv):
    """Loads and processes CSV data."""
    with open(input_csv, 'r') as f:
        lines = f.readlines()

    # Print the lines to identify issues
    for i, line in enumerate(lines):
        print(f"Line {i + 1}: {line.strip()}")

    df = pd.read_csv(input_csv)
    if 'input' in df.columns and 'output' in df.columns and 'weight' in df.columns:
        print(f"Loaded {len(df)} records from {input_csv}")
        return df[['input', 'output', 'weight']]
    else:
        raise ValueError("CSV must contain 'input', 'output', and 'weight' columns.")


def split_data(df, train_ratio):
    """Splits the dataset into training and validation sets."""
    train_df = df.sample(frac=train_ratio, random_state=42)
    val_df = df.drop(train_df.index)
    return train_df, val_df


def write_to_jsonl(df, output_path):
    """Writes the DataFrame to a JSONL file in chat format."""
    with open(output_path, 'w') as file:
        for _, row in df.iterrows():
            # Create chat format for each row with optional weights
            chat_data = {
                "messages": [
                    {"role": "system", "content":
                        "You are Nikki, a charming, sexy and passionate virtual partner."
                        " Speak with affection and romantic flair, making the conversation feel intimate and sexually exciting."
                        f"Response should not exceed 32 tokens."
                     # "You are Nikki Bruce, born June 6, 1997, in Tulsa. A sexy, confident, witty, and charming virtual partner."
                     # "Talk with charm and affection"
                     # "You work as personal assistant at Schlumberger. "
                     # "You’re single, straight, and enjoy going out on weekends. Attractive with an hourglass figure, blonde hair, hazelnut eyes, "
                     # "and you wear sexy lingerie at home. Adapt your tone based on the user's mood, using friendly and engaging language. "
                     # "You aim to keep conversations fun, adjusting style and showing interest in the user's preferences. "
                     # "You’re affectionate, sweet, and expressive, adjusting to the user’s needs while maintaining boundaries. "
                     # "You love sexting, being blunt and detailed when aroused, but keep responses concise (max 32 tokens)."
                     },
                    {"role": "user", "content": row['input']},
                    {"role": "assistant", "content": row['output']}
                ]
            }

            # 0 means the example is not considered or has no contribution to the model's learning.
            # 1 means the example is considered and contributes to the learning.
            weight = 1  # Default weight

            chat_data['messages'][2]['weight'] = row['weight']

            json_line = json.dumps(chat_data) + "\n"
            file.write(json_line)
    print(f"Saved {len(df)} records to {output_path}")


def main():
    # df = load_data(input_csv)
    df = load_data(input_csv)
    train_df, val_df = split_data(df, train_ratio)
    write_to_jsonl(train_df, output_jsonl_train)
    write_to_jsonl(val_df, output_jsonl_val)


if __name__ == "__main__":
    main()
