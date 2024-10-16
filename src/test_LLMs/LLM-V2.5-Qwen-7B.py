from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/Replete-LLM-V2.5-Qwen-7b-GGUF",
	filename="Replete-LLM-V2.5-Qwen-7b-Q4_K_M.gguf",
    n_gpu_layers=-1,
    verbose=False
)

while True:
    text = input("Enter a prompt: ")
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )

    print(response['choices'][0]['message']['content'])
    print("-" * 100)
    print("\n")
    
