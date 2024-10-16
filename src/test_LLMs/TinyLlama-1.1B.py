from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
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
    