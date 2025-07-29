import os

caption_dir = "captions"
user_prompt_dir = "user_prompts"
output_dir = "prompts"

os.makedirs(output_dir, exist_ok=True)

for caption_file in os.listdir(caption_dir):
    if not caption_file.endswith(".txt"):
        continue

    base_name = os.path.splitext(caption_file)[0]
    caption_path = os.path.join(caption_dir, caption_file)
    user_prompt_path = os.path.join(user_prompt_dir, caption_file)
    fused_path = os.path.join(output_dir, caption_file)

    try:
        # Read caption
        with open(caption_path, "r") as f:
            caption = f.read().strip()

        # Read user prompt
        if os.path.exists(user_prompt_path):
            with open(user_prompt_path, "r") as f:
                user_prompt = f.read().strip()
        else:
            user_prompt = ""
            print(f"[⚠️] No user prompt for {base_name}, using caption only.")

        # Combine them
        combined = f"{caption}. {user_prompt}".strip()

        # Save to prompts/
        with open(fused_path, "w") as f:
            f.write(combined)

        print(f"[✅] Combined prompt saved: {fused_path}")

    except Exception as e:
        print(f"[❌] Failed to process {base_name}: {e}")
