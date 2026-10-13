# -*- coding: utf-8 -*-
"""
第 2 步：FastAPI Web 接口。

提供 POST /predict 接口：输入一段文本，返回模型判断结果（好评/差评 + 置信度）。
同时托管前端静态页面（GET / 和 /static/*），并提供健康检查 GET /health。

启动：
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

# 模型 id2label 的英文标签 -> 中文
LABEL_ZH = {
    "negative (stars 1, 2 and 3)": "差评",
    "positive (stars 4 and 5)": "好评",
}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# 模型只加载一次（懒加载 + 线程锁，避免重复加载）
_classifier = None
_lock = threading.Lock()


def get_classifier():
    """加载（或返回已加载的）情感分析 pipeline。"""
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                device = 0 if torch.cuda.is_available() else -1
                _classifier = pipeline(
                    "sentiment-analysis",
                    model=MODEL_NAME,
                    device=device,
                    truncation=True,
                )
    return _classifier


class PredictRequest(BaseModel):
    """接口请求体：一段待分析的评论文本。"""

    text: str = Field(..., min_length=1, max_length=2048, description="待分析的评论文本")


class PredictResponse(BaseModel):
    """接口响应体：标签 + 置信度。"""

    label: str = Field(..., description="中文标签：好评 / 差评")
    label_en: str = Field(..., description="模型原始英文标签")
    confidence: float = Field(..., description="置信度（0~1 的 softmax 概率）")
    model: str = Field(..., description="使用的模型名称")


app = FastAPI(
    title="京东评论中文情感分析 API",
    description="基于 uer/roberta-base-finetuned-jd-binary-chinese 的评论二分类（好评/差评）接口",
    version="1.0.0",
)

# 开发环境放开跨域，方便前端页面直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 托管前端静态资源（页面在 GET / 返回，静态文件挂载到 /static）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """返回前端页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """健康检查。"""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """对输入文本做情感二分类，返回好评/差评与置信度。"""
    try:
        classifier = get_classifier()
        result = classifier(req.text)[0]
        label_en = result["label"]
        return PredictResponse(
            label=LABEL_ZH.get(label_en, label_en),
            label_en=label_en,
            confidence=round(float(result["score"]), 4),
            model=MODEL_NAME,
        )
    except Exception as exc:  # 模型加载/推理失败时返回 500
        raise HTTPException(status_code=500, detail=f"模型推理失败: {exc}") from exc
