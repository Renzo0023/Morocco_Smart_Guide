from huggingface_hub import InferenceClient
client = InferenceClient(model="mistral-small-latest", token="TON_TOKEN")

resp = client.chat.completions.create(
    messages=[{"role": "user", "content": "Bonjour !"}],
    max_tokens=30
)
print(resp.choices[0].message["content"])
