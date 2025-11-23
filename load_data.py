import json
from openai import OpenAI
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets

root = Path(".")  # folder that has train/, test/, TACO.py, etc.

# Train split: concatenate all arrow shards
train_files = sorted((root / "train").glob("data-*.arrow"))
train_datasets = [Dataset.from_file(str(f)) for f in train_files]
train = concatenate_datasets(train_datasets)

# Test split: single arrow shard
test_file = root / "test" / "data-00000-of-00001.arrow"
test = Dataset.from_file(str(test_file))
client = OpenAI()

ds = DatasetDict({
    "train": train,
    "test": test,
})

train_ds = ds["train"]

RUBRICS = """
I will provide you a programming question and possibly a Python code solution. You need to determine whether this
question is challenging to evaluate using traditional unit tests. Apply the following criteria to
identify questions that are hard to evaluate using unit tests:
1. Functions involving randomness or probability:
1) Random number generators; 2) Shuffling algorithms; 3) Probability-based functions
2. Time-dependent functions:
1) Functions that get the current time; 2) Timer functions
3. Functions relying on external resources:
1) Network request functions; 2) File system operations; 3) Database queries
4. Concurrency and multithreading functions:
1) Thread synchronization functions; 2) Concurrent operation functions
5. Hardware-related functions:
1) Device driver functions; 2) Hardware sensor reading functions
6. User interface related functions:
1) Graphics rendering functions; 2) User input processing functions
7. Functions with side effects:
1) Functions modifying global state; 2) Logging functions
8. Cryptography-related functions:
1) Functions generating encryption keys; 2) Certain encryption algorithm implementations
9. Machine learning and adaptive algorithm functions:
1) Model training functions; 2) Neural network backpropagation algorithms; 3) Self-tuning
algorithms
10. Complex mathematical or simulation functions:
1) High-precision floating-point calculations; 2) Physical simulations (e.g., fluid dynamics, particle
collisions); 3) Complex optimization algorithms

For the response, let's think step by step. If the question and answer meet the above criteria, please answer YES;
otherwise, answer NO. Please first give the reason for your judgments, followed by your decision.
Your decision should be in the last line of the reply, which ONLY contains one word: YES or NO.

Output Format:
- REASON: one-sentence justification
- YES or NO
"""

def fulfill_prompt(sample):
    assert "question" in sample
    prompt = "### Programming Question\n" + sample["question"] + "\n\n"
    if "solutions" in sample and len(sample["solutions"]) > 0:
        prompt += "### Solution\n" + sample["solutions"][0] + "\n"
    return prompt

max_samples = 10  # testing purpose only
for sample in iter(train_ds):
    sample["solutions"] = json.loads(sample["solutions"])
    sample["input_output"] = json.loads(sample["input_output"])
    sample["raw_tags"] = eval(sample["raw_tags"])
    sample["tags"] = eval(sample["tags"])
    sample["skill_types"] = eval(sample["skill_types"])
    print(sample)

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=RUBRICS,
        input=fulfill_prompt(sample)
    )
    print(response.output_text)
    breakpoint()
