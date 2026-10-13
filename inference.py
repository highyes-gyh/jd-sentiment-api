# -*- coding: utf-8 -*-
"""
第 1 步：本地跑通模型推理。

用 transformers 的 pipeline 加载京东评论中文情感分析模型，
对输入文本输出「好评 / 差评 + 置信度」。

用法：
    python inference.py "这个手机很好用，性价比高！"
    python inference.py            # 不带参数则使用内置示例
"""

import sys

from transformers import pipeline

MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

# 标签映射：模型 id2label 的英文标签 -> 中文
LABEL_ZH = {
    "negative (stars 1, 2 and 3)": "差评",
    "positive (stars 4 and 5)": "好评",
}


def load_pipeline():
    """加载情感分析 pipeline（模型首次运行会自动下载）。"""
    print(f"正在加载模型 {MODEL_NAME} ...")
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def predict(classifier, text: str) -> dict:
    """对一段文本做情感分类，返回 {label, label_en, confidence}。"""
    result = classifier(text)[0]
    label_en = result["label"]
    return {
        "label": LABEL_ZH.get(label_en, label_en),
        "label_en": label_en,
        "confidence": round(float(result["score"]), 4),
    }


def main() -> None:
    texts = sys.argv[1:] or [
        "这个手机很好用，性价比高！",          # 预期：好评
        "质量太差了，用两天就坏了，差评！",      # 预期：差评
        "东西不错，物流也很快，值得购买。",      # 预期：好评
    ]
    classifier = load_pipeline()
    for text in texts:
        result = predict(classifier, text)
        print(f"文本: {text}")
        print(f"结果: {result['label']}（{result['label_en']}），置信度 {result['confidence']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
