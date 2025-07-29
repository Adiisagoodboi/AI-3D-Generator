import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load model and processor
print("🔄 Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on: {device}")

def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[❌] Error generating caption for {image_path}: {e}")
        return None

def main():
    image_dir = "images"
    user_prompt_dir = "user_prompts"
    caption_dir = "captions"
    output_dir = "prompts"

    os.makedirs(caption_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📁 Found {len(os.listdir(image_dir))} files in 'images/'")

    for file in os.listdir(image_dir):
        if not file.lower().endswith((".jpg", ".png",".jpeg")):
            print(f"[⏩] Skipping non-image file: {file}")
            continue

        base = os.path.splitext(file)[0]
        image_path = os.path.join(image_dir, file)
        user_prompt_path = os.path.join(user_prompt_dir, f"{base}.txt")
        caption_path = os.path.join(caption_dir, f"{base}.txt")
        output_path = os.path.join(output_dir, f"{base}.txt")

        print(f"\n🔍 Processing: {file}")

        # Step 1: Generate caption
        caption = generate_caption(image_path)
        if caption:
            print(f"[📝] Caption: {caption}")
            try:
                with open(caption_path, "w") as f:
                    f.write(caption)
                print(f"[✅] Saved caption to: {caption_path}")
            except Exception as e:
                print(f"[❌] Failed to write caption to file: {e}")
                continue
        else:
            print(f"[❌] Skipping fusion due to missing caption for {file}")
            continue

        # Step 2: Fuse with user prompt
        user_text = ""
        if os.path.exists(user_prompt_path):
            with open(user_prompt_path, "r") as f:
                user_text = f.read().strip()
        else:
            print(f"[⚠️] No user prompt found for: {file}")

        final_prompt = f"{caption}. {user_text}".strip()
        try:
            with open(output_path, "w") as f:
                f.write(final_prompt)
            print(f"[✅] Final prompt saved: {output_path}")
        except Exception as e:
            print(f"[❌] Failed to write final prompt: {e}")

if __name__ == "__main__":
    print("🚀 Running full captioning + fusion pipeline...\n")
    main()
    print("\n✅ All done!")
