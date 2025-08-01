from huggingface_hub import InferenceClient
import os

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
client = InferenceClient(token=HF_API_TOKEN)

client = InferenceClient(
    token=HF_API_TOKEN,
    provider="nebius"
)

import inspect
print(inspect.signature(client.text_generation))


def build_prompt(url, features, pred_label):
    label = "phishing" if pred_label == 1 else "benign"
    return f"""
A user submitted the URL: {url}
It was predicted as **{label}**.

Here are some features:
- Contains suspicious token like 'login' or 'secure': {features['https_token']}
- Uses an IP address: {features['has_ip']}
- Number of dots: {features['nb_dots']}
- URL length: {features['length_url']}

Explain in simple terms why this URL might be considered {label}.
"""

def get_genai_reasoning(prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Error generating explanation:", e)
        import traceback
        traceback.print_exc()
        return f"Error generating explanation: {e}"