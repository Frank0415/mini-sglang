from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
local_dir = "models/smollm2-135m-instruct"

device = "cuda"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint,cache_dir=local_dir)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint,cache_dir=local_dir).to(device)

messages = [{"role": "user", "content": "What is gravity?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(
    inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
)
print(tokenizer.decode(outputs[0]))
