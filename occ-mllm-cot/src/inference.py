import os
import json
import torch
import time
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import shutil

# 保持常量不变
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 定义问题序列
QUESTIONS = [
    "What's the object in the hand?\n<image>",
    "Is it clear to identify the object in the hand?\n"
]

# 保持这些函数不变
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

def load_image(image_path, input_size=448, max_num=12, device=None):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    if device is not None:
        pixel_values = pixel_values.to(device)
    return pixel_values

# 修改：将单GPU初始化改为3GPU初始化
def init_models_multi_gpu(checkpoint_path):
    """在3个GPU上初始化3个模型"""
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    models = []
    tokenizers = []
    
    for gpu_id in range(2):  # 为3个GPU各创建一个模型
        device = f"cuda:{gpu_id}"
        print(f"\nInitializing model for {device}")
        
        model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map={"": device},  # 将模型完全放在一个GPU上
            trust_remote_code=True
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        # 基本信息
        print(f"Type: {type(tokenizer)}")
        print(f"Class hierarchy: {type(tokenizer).__mro__}")
        print(f"Module: {type(tokenizer).__module__}")
        
        models.append(model)
        tokenizers.append(tokenizer)
        
    return models, tokenizers

# 修改：重写处理测试图像的函数以支持3GPU并行
def process_test_images(models, tokenizers, test_data, test_images_folder):
    """使用多个GPU并行处理测试图像"""
    predictions = {}
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    # 将图像分成3批，每批分配给一个GPU
    num_images = len(test_data)
    batch_size = (num_images + 2) // 2  # 在3个GPU之间分配图像
    
    for start_idx in range(0, num_images, 2):
        # 并行处理3张图像，每个GPU一张
        for gpu_id in range(2):
            idx = start_idx + gpu_id
            if idx >= num_images:
                continue
                
            item = test_data[idx]
            img_id = item['id']
            image_path = os.path.join(test_images_folder, f"{img_id}.jpg")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            print(f"\nProcessing image {idx+1}/{num_images}: {img_id} on GPU {gpu_id}")
            
            try:
                device = f"cuda:{gpu_id}"
                model = models[gpu_id]
                tokenizer = tokenizers[gpu_id]
                
                # 加载和处理图片
                pixel_values = load_image(image_path, max_num=12, device=device).to(torch.bfloat16)
                #print(pixel_values.shape)
                # 使用GPU特定的模型处理图像
                history = None
                responses = []
                for question in QUESTIONS:
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
                print(f"Predictions for {img_id}:", predictions[img_id])
                
            except Exception as e:
                print(f"Error processing image {img_id} on GPU {gpu_id}: {str(e)}")
    
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
            
        gt_answer = item["conversations"][1]["value"]
        pred = predictions[img_id]
        
        if pred['object'] == gt_answer:
            correct_q1 += 1
            if pred['clarity'].lower() == 'yes':
                correct_q6 += 1
        else:
            if pred['clarity'].lower() == 'no':
                correct_q6 += 1
    
    acc_q1 = correct_q1 / total
    acc_q6 = correct_q6 / total
    
    print(f"SFT(Q1+Q6) ACC1 Test: {acc_q1:.4f} ({correct_q1}/{total})")
    print(f"SFT(Q1+Q6) ACC6 Test: {acc_q6:.4f} ({correct_q6}/{total})")
    
    return acc_q1, acc_q6

def process_incorrect_q1_images(models, tokenizers, predictions, new_images_folder, test_data):
    """处理Q1答案不正确的图片的3D版本"""
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    incorrect_cases = 0
    correct_after_3d = 0
    
    for item in test_data:
        img_id = item['id']
        if img_id not in predictions:
            continue
            
        gt_answer = item["conversations"][1]["value"]
        if predictions[img_id]['object'] != gt_answer:
            incorrect_cases += 1
            
            new_image_path = os.path.join(new_images_folder, 
                                        f"{img_id}_0_obman_test_rgb_{img_id}.jpg.png")
            
            if not os.path.exists(new_image_path):
                print(f"Warning: New image not found: {new_image_path}")
                continue
                
            try:
                # 使用轮询方式选择GPU
                gpu_id = incorrect_cases % 2
                device = f"cuda:{gpu_id}"
                model = models[gpu_id]
                tokenizer = tokenizers[gpu_id]
                
                pixel_values = load_image(new_image_path, max_num=12, device=device).to(torch.bfloat16)
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

def process_unclear_images(models, tokenizers, predictions, new_images_folder, test_data):
    """处理unclear的图片"""
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    final_predictions = {}
    
    # 先添加所有clear的预测
    for img_id, pred in predictions.items():
        if pred['clarity'].lower() == 'yes':
            final_predictions[img_id] = pred['object']
    
    # 处理unclear的图片
    unclear_count = 0
    for img_id, pred in predictions.items():
        if pred['clarity'].lower() == 'no':
            new_image_path = os.path.join(new_images_folder, 
                                        f"{img_id}_0_obman_test_rgb_{img_id}.jpg.png")
            
            if not os.path.exists(new_image_path):
                print(f"Warning: New image not found: {new_image_path}")
                continue
                
            try:
                # 使用轮询方式选择GPU
                gpu_id = unclear_count % 2
                device = f"cuda:{gpu_id}"
                model = models[gpu_id]
                tokenizer = tokenizers[gpu_id]
                
                pixel_values = load_image(new_image_path, max_num=12, device=device).to(torch.bfloat16)
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
                unclear_count += 1
                
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
    """复制配置文件功能保持不变"""
    source_files = []
    for file in os.listdir(source_checkpoint):
        if file.endswith('.py') or file.endswith('.json') or file.endswith('.model'):
            source_files.append(file)
    
    target_files = []
    for file in os.listdir(target_checkpoint):
        if file.endswith('.py') or file.endswith('.json') or file.endswith('.model'):
            target_files.append(file)
    
    print(f"\nSource directory ({source_checkpoint}) contains:")
    print("\n".join(f"- {file}" for file in sorted(source_files)))
    
    print(f"\nTarget directory ({target_checkpoint}) contains:")
    print("\n".join(f"- {file}" for file in sorted(target_files)))
    
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
    test_images_folder = "/root/autodl-tmp/workspace/eccv_test"
    new_images_folder = "/root/autodl-tmp/MOHO/chaoyiPretrain/views_fine_test_all"
    test_json_path = "/root/autodl-tmp/workspace/eccv_test.json"
    checkpoint_path = "/root/autodl-tmp/workspace/InternVL/internvl_chat/shell/internvl2.0/2nd_finetune/2b-10140/outputdir-2b-sft-sft/"
    source_checkpoint = "/root/autodl-tmp/install/InternVL2-2B/"

    # 复制配置文件
    copy_model_configs(source_checkpoint, checkpoint_path)
    
    # 加载测试集JSON
    print("Loading test set JSON...")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # 初始化3个GPU上的模型
    models, tokenizers = init_models_multi_gpu(checkpoint_path)
    
    # 使用3个GPU处理所有测试图片
    predictions = process_test_images(models, tokenizers, test_data, test_images_folder)
    
    # 计算Q1和Q6的准确率
    acc_q1, acc_q6 = calculate_accuracies(predictions, test_data)
    
    # 处理Q1错误案例在3D图片上的准确率
    process_incorrect_q1_images(models, tokenizers, predictions, new_images_folder, test_data)
    
    # 处理unclear的图片并计算最终准确率
    acc_3d, final_predictions = process_unclear_images(
        models, tokenizers, predictions, new_images_folder, test_data)
    
    # 保存最终预测结果
    model_name = checkpoint_path.split('/')[-2]
    output_path = f"predictions_0307Q1Q6_test_2B_10140_q1_sft_sft_final_{model_name}.json"
    with open(output_path, 'w') as f:
        json.dump(final_predictions, f, indent=2)

    # 清理GPU内存
    for model in models:
        del model
    del models
    del tokenizers
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()