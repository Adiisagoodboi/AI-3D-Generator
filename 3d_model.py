import torch
import os
import gc
import time
from tqdm import tqdm

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def read_prompts_from_folder(folder_path):
    prompts = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin1") as f:
                    text = f.read().strip()
            prompts.append(text)
            filenames.append(os.path.splitext(filename)[0])
    return prompts, filenames

def generate_mesh(prompt, fname, output_dir, device, prior, transmitter, diffusion):
    ply_path = os.path.join(output_dir, f"{fname}_0.ply")
    obj_path = os.path.join(output_dir, f"{fname}_0.obj")

    latents = sample_latents(
        batch_size=1,
        model=prior,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(texts=[prompt]),
        progress=True,
        clip_denoised=True,
        use_fp16=False,
        use_karras=True,
        karras_steps=32,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    for j, latent in enumerate(latents):
        mesh = decode_latent_mesh(transmitter, latent).tri_mesh()

        with open(ply_path, "wb") as f:
            mesh.write_ply(f)

        with open(obj_path, "w") as f:
            mesh.write_obj(f)

        print(f"✅ Saved: {ply_path}, {obj_path}")

def main():
    # Paths
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prompt_folder = os.path.join(base_path, "prompts")
    output_dir = os.path.join(base_path, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Read prompts
    prompts, filenames = read_prompts_from_folder(prompt_folder)
    if not prompts:
        print("⚠️ No prompt files found in 'prompts' folder.")
        return

    # Force CPU
    device = torch.device("cpu")
    print(f"🧠 Running on: {device}")

    # Load models
    print("🔄 Loading models...")
    prior = load_model("text300M", device=device)
    transmitter = load_model("transmitter", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    print(f"📂 Found {len(prompts)} prompt(s). Generating 3D models...\n")

    total_time = 0.0
    success_count = 0
    fail_count = 0

    for i, (prompt, fname) in enumerate(tqdm(zip(prompts, filenames), total=len(prompts), desc="Progress")):
        ply_path = os.path.join(output_dir, f"{fname}_0.ply")
        obj_path = os.path.join(output_dir, f"{fname}_0.obj")

        # ✅ Skip if both output files already exist
        if os.path.exists(ply_path) and os.path.exists(obj_path):
            print(f"⏭️ Skipping '{fname}' — already generated.")
            continue
        
        try:
            print(f"\n📝 [{i+1}] Prompt: '{prompt}'")
            start = time.time()
            generate_mesh(prompt, fname, output_dir, device, prior, transmitter, diffusion)
            end = time.time()
            elapsed = end - start
            print(f"⏱️ Time taken: {elapsed:.2f} sec\n")
            total_time += elapsed
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to generate for prompt '{fname}': {e}")
            fail_count += 1
        finally:
            gc.collect()

    # Final Summary
    print("\n📊 Experiment Summary:")
    print(f"✅ Total successful generations: {success_count}")
    print(f"❌ Total failed generations: {fail_count}")
    print(f"⏳ Total time taken: {total_time:.2f} seconds")
    if success_count > 0:
        print(f"⏱️ Average time per model: {total_time / success_count:.2f} seconds")

if __name__ == "__main__":
    main()
