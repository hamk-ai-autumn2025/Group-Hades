import sys
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ==========================
# CONFIGURATION
# ==========================

# Model to use
MODEL_NAME = "tiiuae/falcon-7b-instruct"  # Falcon 7B Instruct

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
print(f"Loading model {MODEL_NAME} on {DEVICE} ... This may take a minute.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)

# ==========================
# FUNCTIONS
# ==========================

def generate_variant(prompt, profile_desc, max_new_tokens=200, temperature=0.8, top_p=0.95):
    """Generate one variant using a profile description."""
    input_text = f"Profile: {profile_desc}\nPrompt: {prompt}\nGenerate a catchy meme caption or idea:"
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove the input prompt from the output
    return output_text.replace(input_text, "").strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python memeseo.py \"<your meme idea>\"")
        sys.exit(1)

    seed_prompt = sys.argv[1]

    profiles = {
        "high_creativity": "High creativity, playful, attention-grabbing, abundant synonyms and punchy caption.",
        "balanced": "Balanced creativity and SEO—concise, on-brand copy with keyword density.",
        "seo_heavy": "SEO-first: maximize synonyms, long-tail keywords, meta copy, and keyword-rich variations."
    }

    results = {
        "seed_prompt": seed_prompt,
        "model_default": MODEL_NAME,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": []
    }

    for i, (profile, desc) in enumerate(profiles.items(), 1):
        print(f"\nGenerating variant {i}/3 using profile '{profile}' ({desc}) ...")
        variant = generate_variant(seed_prompt, desc)
        results["variants"].append({
            "profile": profile,
            "output": variant
        })
        print(f"\n🧠 Output:\n{variant}\n")

    print("\n=== Full JSON output ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
