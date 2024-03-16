import gradio as gr
import torch
import transformers

# Assuming DEVICE is already defined (e.g., 'cuda' or 'cpu')
DEVICE = 'cuda'  # or 'cpu' if you are not using CUDA

MODEL_DIR = '/notebooks/naifu/checkopint/checkpoint-e4_s12500'

model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
model.to(DEVICE)

def generate_text(prompt, max_length=20):
    """Generates text based on the input prompt."""
    model.eval()  # Set the model to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    attention_mask = torch.ones(input_ids.shape, device=DEVICE)  # Create an attention mask for the inputs
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
    )
    
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find(tokenizer.eos_token)] if tokenizer.eos_token else text  # Remove the end of sequence token

    # Return the generated text
    return text

# Define Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[gr.Textbox(lines=2, placeholder="Enter your prompt here..."), gr.Slider(minimum=10, maximum=300, value=50)],
    outputs=gr.Textbox(label="Generated Text"),
    title="GPT-2 Text Generation",
    description="This model generates text based on the input prompt. It's fine-tuned from GPT-2."
)

# Launch the interface
iface.launch(share=True)