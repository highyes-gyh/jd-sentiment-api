# 京东评论中文情感分析 Web 应用

一个小作业项目：用开源小模型做中文情感分析（京东评论二分类：**好评 / 差评**），
用 FastAPI 提供 Web 接口，配一个纯 HTML 前端页面，全程 Git 管理并推送到 GitHub。

## 技术栈

| 组件   | 说明                                                        |
| ------ | ----------------------------------------------------------- |
| 模型   | [uer/roberta-base-finetuned-jd-binary-chinese](https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese)（RoBERTa-base，1.02 亿参数） |
| 推理   | HuggingFace `transformers` pipeline（sentiment-analysis）     |
| 后端   | FastAPI + Uvicorn                                            |
| 前端   | 纯 HTML + 原生 JavaScript（无任何框架）                        |
| 版本管理 | Git（本地分步提交，最终推送 GitHub）                          |

## 模型说明

- 模型在京东评论数据集上微调，二分类任务；
- 标签映射：`0 → negative (stars 1, 2 and 3)` = **差评**，`1 → positive (stars 4 and 5)` = **好评**；
- 输入一段中文评论，输出类别与置信度（softmax 概率）。

## 项目结构

```
jd-sentiment-api/
├── README.md            # 项目说明
├── requirements.txt     # 依赖清单
├── inference.py         # 第 1 步：本地跑通模型推理
├── main.py              # 第 2 步：FastAPI 接口
├── test_api.py          # 接口自测脚本
└── static/
    └── index.html       # 第 3 步：前端页面
```

## 本地运行

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate        # Windows（Linux/macOS: source .venv/bin/activate）
pip install -r requirements.txt

# 2. 本地跑通一次推理（首次运行会自动下载模型，约 400MB）
python inference.py "这个手机很好用，性价比高！"

# 3. 启动 Web 服务
uvicorn main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 <http://127.0.0.1:8000> ，在输入框里写一句评论，点按钮即可看到结果。

## 接口说明

### POST /predict

请求体（JSON）：

```json
{ "text": "东西不错，物流也快" }
```

响应体（JSON）：

```json
{
  "label": "好评",
  "label_en": "positive (stars 4 and 5)",
  "confidence": 0.9782,
  "model": "uer/roberta-base-finetuned-jd-binary-chinese"
}
```

命令行测试：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"质量太差了，用两天就坏了\"}"
```

### GET /health

服务健康检查，返回 `{"status": "ok"}`。
