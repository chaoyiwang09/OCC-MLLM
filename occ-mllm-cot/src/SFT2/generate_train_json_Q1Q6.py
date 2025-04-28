import json
import jsonlines

# Read the wrong IDs from the first JSON file
with open('/root/autodl-tmp/workspace/obman_train_checkpoint-30000_wrong_1_moho.json', 'r') as f:
    wrong_ids = set(item['answer'] for item in json.load(f))

# Read and process the JSONL file
output_data = []
with jsonlines.open('/root/autodl-tmp/workspace/eccv_train_convert3.jsonl', 'r') as reader:
    for item in reader:
        # Create new item with id and image
        new_item = {
            'id': item['id'],
            'image': item['image']
        }
        
        # Get only the first Q&A pair from original conversations
        first_qa_pair = item['conversations'][:2]  # 获取前两个元素（第一对问答）
        
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
        
        output_data.append(new_item)

# Write the modified data to a new JSONL file
with jsonlines.open('/root/autodl-tmp/workspace/eccv_train_convert6.jsonl', 'w') as writer:
    writer.write_all(output_data)