import json

import laya
import torch


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable inside the container")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")

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
