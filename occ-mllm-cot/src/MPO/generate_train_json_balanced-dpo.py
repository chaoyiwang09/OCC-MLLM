import json

def convert_to_dpo_format(input_file, output_file):
    dpo_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 只处理包含明确性问题的对话
            if len(data['conversations']) >= 4:  # 确保有足够的对话轮次
                clarity_qa = data['conversations'][2:4]  # 获取明确性问题的QA对
                if clarity_qa[0]['value'] == "Is it clear to identify the object in the hand?":
                    # 获取GPT的回答
                    gpt_answer = clarity_qa[1]['value']
                    
                    # 确定选择和拒绝的答案
                    if gpt_answer == "Yes.":
                        chosen = "Yes."
                        rejected = "No."
                    elif gpt_answer == "No.":
                        chosen = "No."
                        rejected = "Yes."
                    else:
                        continue  # 如果回答不是"Yes."或"No."，跳过该样本
                    
                    # 创建DPO格式的数据
                    dpo_sample = {
                        "image": data["image"],
                        #"id": data["id"],
                        # 构建prompt，包含前文对话和当前问题
                        "question": (f"{clarity_qa[0]['value']}\n"),
                        "chosen": chosen,  # 期望的回答
                        "rejected": rejected   # 不期望的回答
                    }
                    dpo_data.append(dpo_sample)
    
    # 写入新的jsonl文件
    print(len(dpo_data))
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(dpo_data)} samples to DPO format")

if __name__ == "__main__":
    input_file = "eccv_train_convert6_balanced.jsonl"
    output_file = "eccv_train_convert7_balanced_dpo_train.jsonl"
    convert_to_dpo_format(input_file, output_file)
