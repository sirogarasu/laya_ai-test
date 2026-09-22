import json
import time

import httpx

import laya
import torch


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable inside the container")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
def is_connection_error(error):
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, httpx.TransportError):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


for attempt in range(1, 5):
    try:
        agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")
        break
    except Exception as error:
        if not is_connection_error(error):
            raise
        if attempt == 4:
            raise RuntimeError(
                "Hugging Face からモデルを取得できませんでした。"
                "コンテナから huggingface.co に接続できるか確認してください。"
            ) from error
        delay = 2 ** attempt
        print(f"Model download connection failed; retrying in {delay}s ({attempt}/4)", flush=True)
        time.sleep(delay)

result = agent.predict(
    "二重に請求されました。重複分を返金してください。",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": ["billing", "technical", "sales"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        },
    },
)

print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
