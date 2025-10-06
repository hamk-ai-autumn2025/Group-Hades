import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "tiiuae/falcon-7b-instruct"
DEVICE = "cpu"  # CPU only
MAX_TOKENS = 50  # shorter output for faster CPU generation
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
NUM_VARIANTS = 3

# ------------------------
# PROFILES
# ------------------------
PROFILES = {
    "high_creativity": "High creativity, playful, attention-grabbing, abundant synonyms and punchy caption.",
    "balanced": "Balanced creativity and SEO—concise, on-brand copy with keyword density.",
    "seo_heavy": "SEO-first: maximize synonyms, long-tail keywords, meta copy, and keyword-rich variations."
}

# ------------------------
# LOAD MODEL & TOKENIZER
# ------------------------
print(f"Loading model {MODEL_NAME} on {DEVICE} ... This may take a minute.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32  # CPU-friendly
).to(DEVICE)

# ------------------------
# PROMPT
# ------------------------
if len(sys.argv) < 2:
    print("Usage: python memeseo.py 'funny meme idea'")
    sys.exit(1)

seed_prompt = sys.argv[1]
print(f"\nSeed prompt: {seed_prompt}\n")

# ------------------------
# GENERATE VARIANTS
# ------------------------
variants_output = []

for i, (profile, desc) in enumerate(PROFILES.items(), 1):
    print(f"Generating variant {i}/{NUM_VARIANTS} using profile '{profile}' ({desc}) ...")
    full_prompt = f"{seed_prompt}\nStyle/Instruction: {desc}"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    # Generate output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    variants_output.append({"profile": profile, "output": output_text})

    print(f"\n🧠 Output:\n{output_text}\n")

# ------------------------
# PRINT FULL JSON
# ------------------------
import json
full_result = {
    "seed_prompt": seed_prompt,
    "model_default": MODEL_NAME,
    "generated_at": torch.tensor([torch.randint(0, 1, (1,))]).item(),
    "variants": variants_output
}

print("=== Full JSON output ===")
print(json.dumps(full_result, indent=4))
