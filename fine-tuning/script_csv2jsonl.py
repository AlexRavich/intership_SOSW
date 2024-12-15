import csv
import json

csv_path = "raw_sample_data.csv"
jsonl_path = "raw_sample_data.jsonl"


def convert_csv2jsonl(csv_path, jsonl_path):
    """
    Convert CSV file to JSONL file.
    :param csv_path:
    :param jsonl_path:
    """
    with open(csv_path, mode='r', encoding='utf-8') as csv_file, \
            open(jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for r in reader:
            jsonl_line = json.dumps(r)
            jsonl_file.write(jsonl_line + "\n")

    print(f"The file was successfully converted in {jsonl_path}")


if __name__ == "__main__":
    convert_csv2jsonl(csv_path, jsonl_path)
