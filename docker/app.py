import time

import gradio as gr
import httpx
import laya
import torch

SCHEMA = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": ["billing", "technical", "sales"],
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer request a refund?",
    },
}


def is_connection_error(error):
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, httpx.TransportError):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def load_agent():
    for attempt in range(1, 5):
        try:
            return laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda")
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


def predict(message):
    if not message or not message.strip():
        raise gr.Error("問い合わせ文を入力してください。")
    answers = agent.predict(message.strip(), SCHEMA)["answers"]
    department = answers["department"]
    refund = answers["refund"]
    summary = (
        f"担当部署: {department['choice']} "
        f"（信頼度 {department['confidence']:.1%}）\n"
        f"返金希望スコア: {refund['noul']:.1%}"
    )
    return summary, answers


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable inside the container")
print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
agent = load_agent()

with gr.Blocks(title="Laya 問い合わせ判定") as demo:
    gr.Markdown("# Laya 問い合わせ判定\n文章を入力すると、担当部署と返金希望を判定します。")
    message = gr.Textbox(
        label="問い合わせ文",
        lines=5,
        placeholder="例: 二重に請求されました。重複分を返金してください。",
    )
    run = gr.Button("判定")
    summary = gr.Textbox(label="結果", lines=2)
    details = gr.JSON(label="詳細")
    run.click(predict, inputs=message, outputs=[summary, details])
    message.submit(predict, inputs=message, outputs=[summary, details])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
