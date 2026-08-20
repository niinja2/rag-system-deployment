import json
from pathlib import Path

source = Path(r"D:\msmarco\msmarco\corpus.jsonl")
output = Path(r"C:\Projects\rag-system-deployment\data\processed\msmarco_100k.jsonl")

output.parent.mkdir(parents=True, exist_ok=True)

with source.open(encoding="utf-8") as source_file, \
     output.open("w", encoding="utf-8") as output_file:

    for number, line in enumerate(source_file):
        if number >= 100_000:
            break

        document = json.loads(line)

        record = {
            "doc_id": document["_id"],
            "text": document["text"],
        }

        output_file.write(json.dumps(record) + "\n")

print("Created:", output)