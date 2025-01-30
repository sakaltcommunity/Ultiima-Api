from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import json

# Qwen2.5-7B-Instructモデルとトークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# テキスト生成のパイプラインを作成
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

def handler(request):
    try:
        # リクエストから入力テキストを取得
        input_text = request.json().get("input", "")

        # リクエストパラメータから生成の設定を取得（デフォルト値を設定）
        max_new_tokens = request.json().get("max_new_tokens", 50)
        top_p = request.json().get("top_p", 1.0)
        top_k = request.json().get("top_k", 50)
        temperature = request.json().get("temperature", 1.0)
        repetition_penalty = request.json().get("repetition_penalty", 1.0)

        # モデルによるテキスト生成
        result = generator(
            input_text,
            max_length=max_new_tokens + len(input_text.split()),  # 最大長を設定
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        # 結果を返す
        return {
            "statusCode": 200,
            "body": json.dumps({
                "generated_text": result[0]["generated_text"]
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
