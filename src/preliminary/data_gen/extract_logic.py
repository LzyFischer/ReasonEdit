import json
from pathlib import Path

# your three dataset files
files = [
        "data/logic/level_1.json", 
        # "data/logic/level_2_1.json", 
        # "data/logic/level_3_1.json"
        ]

all_questions = []

for f in files:
    with open(f, "r") as fp:
        data = json.load(fp)
    for entry in data:
        if "question" in entry and entry["question"]:
            q_nl = entry["question"][0].get("<nl>")
            # \n -> \\n
            q_nl = q_nl.replace("\n", "\\n") if q_nl else None
            if q_nl:
                all_questions.append(q_nl)

# save to a text file
# \n
out_path = Path("level_1.txt")
with open(out_path, "w") as f:
    for q in all_questions:
        f.write(q.strip() + "\n\n")

print(f"Extracted {len(all_questions)} question NLs to {out_path}")