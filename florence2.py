import os
import glob
from pathlib import Path
from PIL import Image
import torch
import lightning as pl
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoProcessor

@torch.no_grad()
def main():
    # ==============================
    # Initialize Distributed Training
    # ==============================
    fabric = pl.Fabric(
        num_nodes=os.environ.get("LEPTON_JOB_TOTAL_WORKERS", 1), 
        precision="bf16-mixed"
    )
    fabric.launch()
    fabric.seed_everything(42)
    
    model = AutoModelForCausalLM.from_pretrained("gokaygokay/Florence-2-SD3-Captioner", trust_remote_code=True).to(fabric.device).eval()
    processor = AutoProcessor.from_pretrained("gokaygokay/Florence-2-SD3-Captioner", trust_remote_code=True)
    
    def process_batch(task_prompt, text_input, images):
        prompts = [task_prompt + text_input] * len(images)
        
        # Ensure all images are in RGB mode
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        
        inputs = processor(text=prompts, images=rgb_images, return_tensors="pt").to(fabric.device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3
        )
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        parsed_answers = [processor.post_process_generation(text, task=task_prompt, image_size=(img.width, img.height)) 
                          for text, img in zip(generated_texts, images)]
        return parsed_answers
    
    image_paths = list(glob.glob("/storage/training/nyanko/special-groups/**/*.webp", recursive=True))
    image_paths = image_paths[fabric.global_rank::fabric.world_size]
    
    batch_size = 16  # Adjust this based on your GPU memory
    for i in trange(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(path) for path in batch_paths]
        
        results = process_batch("<DESCRIPTION>", "Describe this image in brief detail.", batch_images)
        
        for path, result in zip(batch_paths, results):
            result = result['<DESCRIPTION>'].replace("An animated image of ", "").replace("Captured from a low-angle perspective", "").replace("Captured from a high-angle perspective", "")\
                .replace("cartoon", "").replace("women", "girl").replace("Japanese", "").replace("woman", "girl").replace("female", "girl").replace("an Asian", "a").strip(",").strip()
                
            txtpath = Path(path).with_suffix(".floerence2.txt")
            with open(txtpath, "w") as f:
                f.write(result)
            
            # print(path, result)

if __name__ == "__main__":
    main()