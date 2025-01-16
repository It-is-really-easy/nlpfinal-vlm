import os
import random
import json
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from model.model import MMultiModal, LanguageConfig, VisualConfig, MultiModalConfig
from qwen.qwen_generation_utils import make_context
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


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


# 随机选择一对图片
def select_random_image_pair(test_dir):
    filenames = os.listdir(os.path.join(test_dir, "A"))
    random_filename = random.choice(filenames)
    image_a_path = os.path.join(test_dir, "A", random_filename)
    image_b_path = os.path.join(test_dir, "B", random_filename)
    return image_a_path, image_b_path, random_filename


# 加载参考描述
def get_reference_description(captions_file, filename):
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    print("filename", filename)
    # 根据文件名找到对应的描述信息
    image_info = next(item for item in captions_data['images'] if item['filename'] == filename)
    captions = [sentence['raw'] for sentence in image_info['sentences']]

    # 随机选择一条描述作为参考答案
    selected_idx = np.random.choice(len(captions))
    return captions[selected_idx]


# 计算METEOR和ROUGE-L指标
def calculate_metrics(prediction, reference):
    # 分词处理
    prediction_tokens = word_tokenize(prediction)
    reference_tokens = word_tokenize(reference)

    # 计算METEOR
    meteor_score_value = meteor_score([reference_tokens], prediction_tokens) * 100  # 转为百分比

    # 计算ROUGE-L
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_score = rouge_scorer_obj.score(' '.join(reference_tokens), ' '.join(prediction_tokens))[
                        'rougeL'].fmeasure * 100  # 转为百分比

    return meteor_score_value, rouge_l_score


# 主函数
def main():
    # 配置模型路径
    base_language_model = "./Qwen-7B-Chat"
    base_value_model = "./clip-vit-large-patch14"

    tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")

    model = MMultiModal(
        LanguageConfig(model_path=base_language_model),
        VisualConfig(model_path=base_value_model),
        MultiModalConfig(replace_token_id=replace_token_id),
        train=False
    ).cuda()
    model.load("./weights/train_V1_5/checkpoint-16000/")
    prompt = "describe the changes between the images<|extra_0|>"

    test_dir = "./dataset/images/test"
    captions_file = "./dataset/LevirCCcaptions.json"  # 标注描述文件路径

    # 随机选择一组图片
    image_a_path, image_b_path, filename = select_random_image_pair(test_dir)

    # 加载并处理图像
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    image_a_pt = image_process(image_a).unsqueeze(0).cuda().to(torch.bfloat16)
    image_b_pt = image_process(image_b).unsqueeze(0).cuda().to(torch.bfloat16)
    images_pt = torch.cat([image_a_pt, image_b_pt], dim=0)

    # 获取参考描述
    reference_description = get_reference_description(captions_file, filename)

    # 构建上下文和生成描述
    raw_text, context_tokens = make_context(
        tokenizer,
        "用英文回答：" + prompt,
        history=[],
        system="你是一位图像理解助手。"
    )
    question_ids = tokenizer.encode(raw_text)
    result = model.generate(images_pt, question_ids)
    generated_description = tokenizer.decode(result[0]).strip()

    # 计算评价指标
    meteor_avg, rouge_l_avg = calculate_metrics(generated_description, reference_description)

    # 绘制图像和描述
    plt.figure(figsize=(12, 6))

    # 显示第一张图片
    plt.subplot(1, 2, 1)
    plt.imshow(image_a)
    plt.axis("off")
    plt.title("Image A")
    # 显示第二张图片
    plt.subplot(1, 2, 2)
    plt.imshow(image_b)
    plt.axis("off")
    plt.title("Image B")

    print("Generated Description:", generated_description)
    print("Reference Description:", reference_description)
    print(f"METEOR(%): {meteor_avg:.2f}")
    print(f"ROUGE-L(%): {rouge_l_avg:.2f}")

    # 在图像下方显示描述
    plt.suptitle(f"Filename: {filename}\n\n"
                 f"Generated Description:\n{generated_description}\n\n"
                 f"Reference Description:\n{reference_description}",
                 fontsize=10, y=0.02)

    # 保存或展示图像
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
