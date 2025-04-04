from transformers import GPT2LMHeadModel, GPT2Tokenizer 
# Load pre-trained model and tokenizer 
model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name) 
model = GPT2LMHeadModel.from_pretrained(model_name) 
# Input text 
input_text = "Deep learning is" 
# Tokenize input 
input_ids = tokenizer.encode(input_text, return_tensors='pt') 
# Generate text 
output = model.generate(input_ids, max_length=100, temperature=0.5, top_k=40,num_return_sequences=1) 
# Decode and print 
generated_text = tokenizer.decode(output[0], skip_special_tokens=True) 
print(generated_text) 
