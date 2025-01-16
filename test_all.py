import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import nltk
# nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from model.model import MMultiModal, LanguageConfig, VisualConfig, MultiModalConfig
from qwen.qwen_generation_utils import make_context


# 图像预处理
def image_process(image):
    mean = [0.485, 0.456, 0.406]  # RGB
    std = [0.229, 0.224, 0.225]  # RGB
    tran = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tran(image)


# 根据文件名获取参考描述
def get_reference_description(captions_file, filename):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)

    normalized_filename = filename.strip().lower()

    try:
        image_info = next(
            item for item in captions_data['images'] if item['filename'].strip().lower() == normalized_filename
        )
        captions = [sentence['raw'] for sentence in image_info['sentences']]
        selected_idx = np.random.choice(len(captions))
        return captions[selected_idx]
    except StopIteration:
        print(f"Warning: No captions found for {filename}. Skipping this image.")
        return None


# 评价指标计算
def calculate_metrics(predictions, references):
    meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(pred)) for pred, ref in zip(predictions, references)]
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [rouge_scorer_obj.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)]
    meteor_avg = sum(meteor_scores) / len(meteor_scores) * 100  # 转为百分比
    rouge_l_avg = sum(rouge_l_scores) / len(rouge_l_scores) * 100  # 转为百分比
    return meteor_avg, rouge_l_avg


# 检查并保存生成结果
def save_results(result_file, filename, result):
    if not os.path.exists(result_file):
        data = {}
    else:
        with open(result_file, 'r') as f:
            data = json.load(f)
    data[filename] = result
    with open(result_file, 'w') as f:
        json.dump(data, f)


# 读取已保存结果
def load_results(result_file):
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return {}


# 主函数
def main():
    # 配置模型路径
    base_language_model = "./Qwen-7B-Chat"
    base_value_model = "./clip-vit-large-patch14"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")

    # 加载模型
    model = MMultiModal(
        LanguageConfig(model_path=base_language_model),
        VisualConfig(model_path=base_value_model),
        MultiModalConfig(replace_token_id=replace_token_id),
        train=False
    ).cuda()
    model.load("./weights/train_V1_5/checkpoint-6000/")

    prompt = "describe the changes between the images<|extra_0|>"

    # 配置路径
    test_dir = "./dataset/images/test"
    captions_file = "./dataset/LevirCCcaptions.json"
    result_file = "./result.json"

    # 加载已有结果
    results = load_results(result_file)

    predictions = []
    references = []
    filenames = os.listdir(os.path.join(test_dir, "A"))

    for filename in tqdm(filenames, desc="Evaluating"):
        image_a_path = os.path.join(test_dir, "A", filename)
        image_b_path = os.path.join(test_dir, "B", filename)

        # 跳过无效图片文件
        if not os.path.exists(image_a_path) or not os.path.exists(image_b_path):
            print(f"Skipping invalid image pair: {filename}")
            continue

        # 获取参考描述
        reference = get_reference_description(captions_file, filename)
        if reference is None:
            continue  # 跳过没有参考描述的图片
        references.append(reference)

        # 如果已有生成结果，直接使用
        if filename in results:
            predictions.append(results[filename])
            continue

        # 图像处理
        image_a = Image.open(image_a_path).convert("RGB")
        image_b = Image.open(image_b_path).convert("RGB")
        image_a_pt = image_process(image_a).unsqueeze(0).cuda().to(torch.bfloat16)
        image_b_pt = image_process(image_b).unsqueeze(0).cuda().to(torch.bfloat16)
        images_pt = torch.cat([image_a_pt, image_b_pt], dim=0)

        # 模型生成预测描述
        raw_text, context_tokens = make_context(
            tokenizer,
            "用英文回答：" + prompt,
            history=[],
            system="你是一位图像理解助手。"
        )
        question_ids = tokenizer.encode(raw_text)
        result = model.generate(images_pt, question_ids)
        result_text = tokenizer.decode(result[0]).strip()

        # 保存生成结果
        save_results(result_file, filename, result_text)
        predictions.append(result_text)

    # 计算评价指标
    if not predictions or not references:
        raise ValueError("No valid predictions or references. Please check your dataset.")

    meteor_avg, rouge_l_avg = calculate_metrics(predictions, references)
    print(f"METEOR(%): {meteor_avg:.2f}")
    print(f"ROUGE-L(%): {rouge_l_avg:.2f}")


if __name__ == "__main__":
    main()
