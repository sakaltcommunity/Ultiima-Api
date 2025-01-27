from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

MODEL_NAME = "Sakalti/ultiima-78B"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@app.post("/api-sakal-models")
async def generate_text(prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
    )
    return {"generated_text": tokenizer.decode(output[0], skip_special_tokens=True)}
