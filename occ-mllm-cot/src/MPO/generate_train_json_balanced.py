import json
import json
import jsonlines
import random

# Read the wrong IDs from the first JSON file
with open('/root/autodl-tmp/workspace/obman_train_checkpoint-30000_wrong_1_moho.json', 'r') as f:
    wrong_ids = set(item['answer'] for item in json.load(f))

# Read and process the JSONL file - first pass to separate positive and negative samples
positive_samples = []
negative_samples = []

with jsonlines.open('/root/autodl-tmp/workspace/eccv_train_convert3.jsonl', 'r') as reader:
    for item in reader:
        # Create new item with id and image
        new_item = {
            'id': item['id'],
            'image': item['image']
        }
        
        # Get only the first Q&A pair from original conversations
        first_qa_pair = item['conversations'][:2]
        
        # Create new Q&A pair
        new_question = {
            "from": "human",
            "value": "Is it clear to identify the object in the hand?"
        }
        
        new_answer = {
            "from": "gpt",
            "value": "No." if item['id'] in wrong_ids else "Yes."
        }
        
        # Combine first Q&A pair with new Q&A pair
        new_item['conversations'] = first_qa_pair + [new_question, new_answer]
        
        # Separate into positive and negative samples
        if item['id'] in wrong_ids:
            negative_samples.append(new_item)
        else:
            positive_samples.append(new_item)

# Get the minimum length between positive and negative samples
min_length = min(len(positive_samples), len(negative_samples))

# Randomly sample from the larger group to match the size of the smaller group
if len(positive_samples) > min_length:
    positive_samples = random.sample(positive_samples, min_length)
elif len(negative_samples) > min_length:
    negative_samples = random.sample(negative_samples, min_length)

# Combine the balanced samples
balanced_data = positive_samples + negative_samples

# Shuffle the final data
random.shuffle(balanced_data)

# Write the balanced data to a new JSONL file
with jsonlines.open('/root/autodl-tmp/workspace/eccv_train_convert6_balanced.jsonl', 'w') as writer:
    writer.write_all(balanced_data)

# Print statistics
print(f"Number of positive samples: {len(positive_samples)}")
print(f"Number of negative samples: {len(negative_samples)}")
print(f"Total samples in balanced dataset: {len(balanced_data)}")