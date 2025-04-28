import os
import json
import torch
import time
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import shutil


# 图像预处理相关常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """构建图像转换pipeline"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """找到最接近的宽高比"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """动态预处理图像"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def load_image(image_path, input_size=448, max_num=12):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def init_model_multi_gpu(checkpoint_path):
    """初始化模型到多个GPU"""
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    model = AutoModel.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        device_map='auto',
        trust_remote_code=True
    ).eval()
    
    print("\nModel distribution across devices:")
    for name, device in model.hf_device_map.items():
        print(f"{name}: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    return model, tokenizer

def process_test_images(model, tokenizer, test_data, test_images_folder):
    """处理测试集图片"""
    predictions = {}
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    device = next(model.parameters()).device
    
    for i, item in enumerate(test_data):
        img_id = item['id']
        image_path = os.path.join(test_images_folder, f"{img_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
        print(f"\nProcessing image {i+1}/{len(test_data)}: {img_id}")
        
        try:
            # 加载和处理图片
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device)
            
            # 问两个问题
            questions = [
                "What's the object in the hand?\n<image>",
                "Is it clear to identify the object in the hand??\n"
            ]
            
            responses = []
            history = None
            
            for question in questions:
                response, history = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=history,
                    return_history=True
                )
                responses.append(response)
            
            predictions[img_id] = {
                'object': responses[0],
                'clarity': responses[1].lower().strip().replace(".",'')
            }
            print(predictions[img_id])
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            
    return predictions

def calculate_accuracies(predictions, test_data):
    """计算Q1和Q6的准确率"""
    total = len(predictions)
    correct_q1 = 0
    correct_q6 = 0
    
    for item in test_data:
        img_id = item['id']
        if img_id not in predictions:
            continue
            
        # 获取ground truth答案
        gt_answer = item["conversations"][1]["value"]
        pred = predictions[img_id]
        
        # 计算Q1准确率
        if pred['object'] == gt_answer:
            correct_q1 += 1
            # Q1正确且clarity为yes，Q6正确
            if pred['clarity'].lower() == 'yes':
                correct_q6 += 1
        else:
            # Q1错误且clarity为no，Q6正确
            if pred['clarity'].lower() == 'no':
                correct_q6 += 1
    
    acc_q1 = correct_q1 / total
    acc_q6 = correct_q6 / total
    
    print(f"SFT(Q1+Q6) ACC1 Test: {acc_q1:.4f} ({correct_q1}/{total})")
    print(f"SFT(Q1+Q6) ACC6 Test: {acc_q6:.4f} ({correct_q6}/{total})")
    
    return acc_q1, acc_q6

def process_incorrect_q1_images(model, tokenizer, predictions, new_images_folder, test_data):
    """处理Q1答案不正确的图片的3D版本"""
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    device = next(model.parameters()).device
    incorrect_cases = 0
    correct_after_3d = 0
    
    for item in test_data:
        img_id = item['id']
        if img_id not in predictions:
            continue
            
        gt_answer = item["conversations"][1]["value"]
        # 找出Q1答案不正确的案例
        if predictions[img_id]['object'] != gt_answer:
            incorrect_cases += 1
            
            # 直接使用3D图片进行推理
            new_image_path = os.path.join(new_images_folder, 
                                        f"{img_id}_0_obman_test_rgb_{img_id}.jpg.png")
            
            if not os.path.exists(new_image_path):
                print(f"Warning: New image not found: {new_image_path}")
                continue
                
            try:
                pixel_values = load_image(new_image_path, max_num=12).to(torch.bfloat16).to(device)
                question = "What's the object in the hand?\n<image>"
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True
                )
                
                if response == gt_answer:
                    correct_after_3d += 1
                    
            except Exception as e:
                print(f"Error processing new image {img_id}: {str(e)}")
    
    if incorrect_cases > 0:
        acc_3d_plus = correct_after_3d / incorrect_cases
        print(f"SFT+Hands on 3d plus: {acc_3d_plus:.4f} ({correct_after_3d}/{incorrect_cases})")
    else:
        print("No incorrect Q1 cases found")
    
    return correct_after_3d, incorrect_cases

def process_unclear_images(model, tokenizer, predictions, new_images_folder, test_data):
    """处理unclear的图片"""
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    device = next(model.parameters()).device
    final_predictions = {}
    
    # 先添加所有clear的预测
    for img_id, pred in predictions.items():
        if pred['clarity'].lower() == 'yes':
            final_predictions[img_id] = pred['object']
    
    # 处理unclear的图片
    for img_id, pred in predictions.items():
        if pred['clarity'].lower() == 'no':
            new_image_path = os.path.join(new_images_folder, 
                                        f"{img_id}_0_obman_test_rgb_{img_id}.jpg.png")
            
            if not os.path.exists(new_image_path):
                print(f"Warning: New image not found: {new_image_path}")
                continue
                
            try:
                pixel_values = load_image(new_image_path, max_num=12).to(torch.bfloat16).to(device)
                question = "What's the object in the hand?\n<image>"
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=True
                )
                final_predictions[img_id] = response
                
                # 新功能1：如果新的答案和原来的不一样，打印前后的答案
                #if response != pred['object']:
                #    print(f"\nAnswer changed for image {img_id}:")
                #    print(f"Original answer: {pred['object']}")
                #    print(f"New answer from 3D image: {response}")
                
            except Exception as e:
                print(f"Error processing new image {img_id}: {str(e)}")
    
    # 计算最终准确率
    total = len(test_data)
    correct = 0
    for item in test_data:
        img_id = item['id']
        if img_id in final_predictions:
            gt_answer = item["conversations"][1]["value"]
            if final_predictions[img_id] == gt_answer:
                correct += 1
    
    acc_3d = correct / total
    print(f"SFT(Q1+Q6)Model1 on 3D accuracy: {acc_3d:.4f} ({correct}/{total})")
    
    return acc_3d, final_predictions

def copy_model_configs(source_checkpoint, target_checkpoint):
    """
    复制source_checkpoint中的.py和.json文件到target_checkpoint，
    但仅复制target_checkpoint中不存在的文件
    
    Args:
        source_checkpoint: 源checkpoint路径
        target_checkpoint: 目标checkpoint路径
    """
    # 获取源目录中所有的.py和.json文件
    source_files = []
    for file in os.listdir(source_checkpoint):
        if file.endswith('.py') or file.endswith('.json'):
            source_files.append(file)
    
    # 获取目标目录中所有的.py和.json文件
    target_files = []
    for file in os.listdir(target_checkpoint):
        if file.endswith('.py') or file.endswith('.json'):
            target_files.append(file)
    
    # 打印目录内容
    print(f"\nSource directory ({source_checkpoint}) contains:")
    print("\n".join(f"- {file}" for file in sorted(source_files)))
    
    print(f"\nTarget directory ({target_checkpoint}) contains:")
    print("\n".join(f"- {file}" for file in sorted(target_files)))
    
    # 找出需要复制的文件（在源目录中存在但在目标目录中不存在的文件）
    files_to_copy = [f for f in source_files if f not in target_files]
    
    if files_to_copy:
        print("\nCopying the following files:")
        for file in files_to_copy:
            source_path = os.path.join(source_checkpoint, file)
            target_path = os.path.join(target_checkpoint, file)
            shutil.copy2(source_path, target_path)
            print(f"- Copied: {file}")
    else:
        print("\nNo files need to be copied - target directory already has all necessary files.")

def main():
    # 设置路径
    test_images_folder = "/root/autodl-tmp/obman/test/rgb"
    new_images_folder = "/root/autodl-tmp/MOHO/chaoyiPretrain/views_fine_test_all"
    checkpoint_path = "/root/autodl-tmp/workspace/Base1_Stage1/InternVL/internvl_chat/shell/internvl2.0_MPO/output_MPO"
    test_json_path = "/root/autodl-tmp/workspace/eccv_test.json"
    source_checkpoint = "/root/autodl-tmp/workspace/install/InternVL2-4B/"
    copy_model_configs(source_checkpoint, checkpoint_path)
    # 加载测试集JSON
    print("Loading test set JSON...")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # 初始化模型
    model, tokenizer = init_model_multi_gpu(checkpoint_path)
    
    # 1. 处理所有测试图片
    predictions = process_test_images(model, tokenizer, test_data, test_images_folder)
    
    # 2&3&4. 计算Q1和Q6的准确率
    acc_q1, acc_q6 = calculate_accuracies(predictions, test_data)

    # 新功能2：计算Q1错误案例在3D图片上的准确率
    process_incorrect_q1_images(model, tokenizer, predictions, new_images_folder, test_data)
    
    # 5. 处理unclear的图片并计算最终准确率
    acc_3d, final_predictions = process_unclear_images(
        model, tokenizer, predictions, new_images_folder, test_data)
    
    # 保存最终预测结果
    output_path = "final_predictions-checkpoint-MPO-1epoch-Q1Q6plus.json"
    with open(output_path, 'w') as f:
        json.dump(final_predictions, f, indent=2)

if __name__ == "__main__":
    main()