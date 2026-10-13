# -*- coding: utf-8 -*-
"""
接口自测脚本：对运行中的服务发起请求，验证各接口行为。

用法（先启动服务）：
    uvicorn main:app --port 8000
    python test_api.py                     # 默认 http://127.0.0.1:8000
    python test_api.py http://127.0.0.1:8000
"""

import sys

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

CASES = [
    ("好评示例", "这个手机很好用，性价比高！", "好评"),
    ("差评示例", "质量太差了，用两天就坏了，差评！", "差评"),
    ("中性偏好评", "东西不错，物流也很快，值得购买。", "好评"),
]


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'通过' if ok else '失败'}] {name} {detail}")


def main():
    passed = 0
    total = 0

    # 1. 健康检查
    total += 1
    resp = requests.get(f"{BASE_URL}/health", timeout=60)
    ok = resp.status_code == 200 and resp.json()["status"] == "ok"
    check("GET /health", ok, str(resp.status_code))
    passed += ok

    # 2. 前端页面
    total += 1
    resp = requests.get(f"{BASE_URL}/", timeout=60)
    ok = resp.status_code == 200 and "分析情感" in resp.text
    check("GET / 返回前端页面", ok, str(resp.status_code))
    passed += ok

    # 3. 情感分析接口
    for name, text, expect in CASES:
        total += 1
        resp = requests.post(
            f"{BASE_URL}/predict", json={"text": text}, timeout=120
        )
        ok = resp.status_code == 200 and resp.json()["label"] == expect
        detail = resp.json() if resp.status_code == 200 else resp.text[:80]
        check(f"POST /predict（{name}）", ok, str(detail))
        passed += ok

    # 4. 空文本应返回 422 参数校验错误
    total += 1
    resp = requests.post(f"{BASE_URL}/predict", json={"text": ""}, timeout=60)
    ok = resp.status_code == 422
    check("空文本返回 422", ok, str(resp.status_code))
    passed += ok

    print(f"\n共 {total} 项，通过 {passed} 项，失败 {total - passed} 项")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
