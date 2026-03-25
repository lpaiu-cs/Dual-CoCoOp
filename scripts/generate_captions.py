import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = {
    "eurosat": {
        "examples": {
            "Forest": "dense clusters of tree canopies forming irregular green patches with minimal built structures.",
            "Sea or Lake": "large, enclosed water bodies with smooth shorelines and surrounding sparse vegetation or barren land.",
            "Residential Buildings": "clusters of buildings arranged in dense blocks with intersecting roads and minimal open fields.",
            "Highway or Road": "long, paved roads with multiple lanes, often bordered by vehicles and surrounded by undeveloped or sparse land.",
        },
        "instruction": (
            'You are a chatbot that receives a label from the EuroSAT dataset and generates a short, '
            'informative description to replace the generic prompt "a photo of" in the CLIP model\'s '
            "text encoder. Your output should be concise, one sentence only, and tailored for satellite "
            "imagery. Focus on land use, spatial layout, or distinct visual features observable from "
            "above. Avoid vague adjectives, stylistic language, or unnecessary length. Highlight "
            "consistent, identifying characteristics of the class."
        ),
    },
    "fgvc_aircraft": {
        "examples": {
            "707-320": "a long four-engine jet with thin swept-back wings and turbojet engines mounted under the wings.",
            "727-200": "a narrow-body trijet with a long fuselage, T-tail, and three rear-mounted engines.",
            "A330-200": "a wide-body twin-engine jet with a rounded nose, long fuselage, and underwing engines on swept-back wings.",
            "An-12": "a four-engine turboprop transport aircraft with high-mounted straight wings and a prominent rear cargo ramp.",
            "Gulfstream V": "a sleek twin-engine business jet with a pointed nose, swept wings, and rear-mounted engines on a narrow fuselage.",
            "Saab 340": "a small twin-propeller commuter aircraft with straight wings, a T-tail, and a short, boxy fuselage.",
            "Yak-42": "a rear-engined trijet with a circular fuselage, straight wings, and a T-tail configuration.",
        },
        "instruction": (
            "You are a chatbot that receives a class name from the FGVC-Aircraft dataset and generates "
            'a short, visually grounded caption to replace the generic phrase "a photo of" in the CLIP '
            "model's text encoder. Create one sentence that captures externally observable and "
            "discriminative features of each aircraft. Focus on wing shape, engine count and position, "
            "tail design, fuselage layout, and unique proportions or silhouette. Do not include "
            "historical facts, technical specifications, or subjective language. Manufacturer names "
            "should only be included if visually identifying."
        ),
    },
}


def load_labels(dataset_root: Path, dataset: str) -> list[str]:
    if dataset == "eurosat":
        return [
            "Annual Crop Land",
            "Forest",
            "Herbaceous Vegetation Land",
            "Highway or Road",
            "Industrial Buildings",
            "Pasture Land",
            "Permanent Crop Land",
            "Residential Buildings",
            "River",
            "Sea or Lake",
        ]

    if dataset == "fgvc_aircraft":
        variants_path = dataset_root / "fgvc-aircraft-2013b" / "data" / "variants.txt"
        return [line.strip() for line in variants_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    raise ValueError(f"Unsupported dataset: {dataset}")


def output_path(dataset_root: Path, dataset: str) -> Path:
    if dataset == "eurosat":
        return dataset_root / "eurosat" / "eurosat_captions.json"
    if dataset == "fgvc_aircraft":
        return dataset_root / "fgvc-aircraft-2013b" / "data" / "fgvc_aircraft_captions.json"
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_prompt(dataset: str, label: str) -> str:
    prompt_spec = PROMPTS[dataset]
    examples = "\n".join(f"{k} -> {v}" for k, v in prompt_spec["examples"].items())
    return (
        f"{prompt_spec['instruction']}\n\n"
        f"Few-shot examples:\n{examples}\n\n"
        f"Generate a caption for {label}.\n"
        "Return only the caption."
    )


def clean_caption(text: str) -> str:
    caption = text.strip().splitlines()[0].strip()
    if "->" in caption:
        caption = caption.split("->", 1)[1].strip()
    caption = caption.strip('"').strip("'")
    if caption.endswith("</s>"):
        caption = caption[:-4].strip()
    return caption


def generate_caption(tokenizer, model, dataset: str, label: str) -> str:
    messages = [
        {"role": "system", "content": "You follow instructions exactly and return one sentence only."},
        {"role": "user", "content": build_prompt(dataset, label)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    return clean_caption(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(PROMPTS.keys()), required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.dataset_root, args.dataset)
    save_path = output_path(args.dataset_root, args.dataset)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if save_path.exists() and not args.overwrite:
        existing = json.loads(save_path.read_text(encoding="utf-8"))

    pending = [label for label in labels if label not in existing]
    if not pending:
        print(f"All captions already exist at {save_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    captions = dict(existing)
    fixed_examples = PROMPTS[args.dataset]["examples"]

    for idx, label in enumerate(pending, start=1):
        if label in fixed_examples:
            caption = fixed_examples[label]
        else:
            caption = generate_caption(tokenizer, model, args.dataset, label)
        captions[label] = caption
        print(f"[{idx}/{len(pending)}] {label} -> {caption}")

    ordered = {label: captions[label] for label in labels}
    save_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(ordered)} captions to {save_path}")


if __name__ == "__main__":
    main()
