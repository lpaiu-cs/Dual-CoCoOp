import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = {
    "dtd": {
        "examples": {
            "bubbly": "surface covered with small, round protrusions resembling foam or air-filled blisters, giving a light and uneven texture.",
            "frilly": "surface featuring delicate, ruffled edges or layered folds with a soft, fluttering, fabric-like texture.",
            "woven": "interlaced strands or fibers forming a tight, grid-like pattern with a coarse yet structured tactile feel.",
            "crystalline": "surface composed of angular, faceted structures with sharp edges and a rigid, glass-like tactile sensation.",
            "paisley": "surface decorated with intricate, teardrop-shaped motifs arranged in flowing, curved patterns.",
            "polka-dotted": "surface marked with evenly spaced round spots that interrupt a flat background and create a punctuated visual rhythm.",
        },
        "instruction": (
            "You are a chatbot that receives a label from the DTD (Describable Textures Dataset) "
            'and generates a description to replace the fixed prompt "a photo of" in the CLIP '
            "model's text encoder. Focus on texture, material structure, repeated surface patterns, "
            "and tactile sensation. Avoid generic filler, subjective language, or abstract emotional "
            "phrasing. Return one visually grounded sentence only."
        ),
    },
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
    "food101": {
        "examples": {
            "apple_pie": "lattice-topped round pie filled with visible apple slices and a golden-brown crust.",
            "steak": "thick seared meat with a broad brown surface, visible grill marks, and a compact slab-like shape.",
            "tacos": "folded or open tortillas filled with visible meat, lettuce, and chopped toppings.",
            "red_velvet_cake": "layered round cake with deep red sponge and white cream or frosting between layers.",
            "samosa": "triangular fried pastry with a crisp shell and filling visible near the edges.",
        },
        "instruction": (
            "You are a chatbot that extracts concise, externally observable information about a food "
            'label to replace the generic prompt "a photo of" in the CLIP model\'s text encoder. '
            "Focus on visible shape, structure, plating, and prominent ingredients that can be seen "
            "from an image. Avoid subjective quality words, smell, taste, or cultural background. "
            "If a dish has a consistent visual color pattern, you may mention it briefly. Return one "
            "sentence only."
        ),
    },
    "ucf101": {
        "examples": {
            "Tai_Chi": "martial arts movements performed with extended arms and controlled posture in an open or park-like setting.",
            "Trampoline_Jumping": "repetitive vertical leaps performed on a trampoline with extended limbs and airborne motion against a static background.",
            "Biking": "pedaling motion on a bicycle with forward body lean, typically along roads, trails, or open paths.",
            "Breast_Stroke": "swimming action with synchronized arm sweeps and frog-like leg kicks performed horizontally in a pool.",
            "Band_Marching": "synchronized walking in formation while carrying musical instruments, often on open fields or parade grounds.",
            "Apply_Eye_Makeup": "precise hand movements near the eye area using brushes or applicators while facing a mirror.",
        },
        "instruction": (
            "You are a chatbot that receives a label from the UCF101 dataset and generates a short, "
            'informative description to replace the generic prompt "a photo of" in the CLIP model\'s '
            "text encoder. Focus on the nature of the action, body movement, relevant objects, and "
            "typical scene context. Avoid vague adjectives, stylistic language, or unnecessary detail. "
            "Return one sentence only."
        ),
    },
}


def load_labels(dataset_root: Path, dataset: str) -> list[str]:
    if dataset == "dtd":
        split_path = dataset_root / "dtd" / "split_zhou_DescribableTextures.json"
        return load_labels_from_split(split_path)

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

    if dataset == "food101":
        split_path = dataset_root / "food-101" / "split_zhou_Food101.json"
        return load_labels_from_split(split_path)

    if dataset == "ucf101":
        split_path = dataset_root / "ucf101" / "split_zhou_UCF101.json"
        return load_labels_from_split(split_path)

    raise ValueError(f"Unsupported dataset: {dataset}")


def output_path(dataset_root: Path, dataset: str) -> Path:
    if dataset == "dtd":
        return dataset_root / "dtd" / "dtd_captions.json"
    if dataset == "eurosat":
        return dataset_root / "eurosat" / "eurosat_captions.json"
    if dataset == "fgvc_aircraft":
        return dataset_root / "fgvc-aircraft-2013b" / "data" / "fgvc_aircraft_captions.json"
    if dataset == "food101":
        return dataset_root / "food-101" / "food101_captions.json"
    if dataset == "ucf101":
        return dataset_root / "ucf101" / "ucf101_captions.json"
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_labels_from_split(split_path: Path) -> list[str]:
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    classnames = {}
    for split_name in ("train", "val", "test"):
        for _, label, classname in split_data[split_name]:
            classnames[int(label)] = classname
    return [classnames[idx] for idx in sorted(classnames)]


def build_prompt(dataset: str, label: str) -> str:
    prompt_spec = PROMPTS[dataset]
    examples = "\n".join(
        f"{display_label(dataset, key)} -> {value}" for key, value in prompt_spec["examples"].items()
    )
    readable_label = display_label(dataset, label)
    return (
        f"{prompt_spec['instruction']}\n\n"
        f"Few-shot examples:\n{examples}\n\n"
        "Use natural-language text with spaces, not underscores or code-style identifiers.\n\n"
        f'Generate a caption for the class label "{label}" (read as "{readable_label}").\n'
        "Return only the caption."
    )


def clean_caption(text: str) -> str:
    caption = text.strip().splitlines()[0].strip()
    if "->" in caption:
        caption = caption.split("->", 1)[1].strip()
    caption = caption.strip('"').strip("'")
    if caption.endswith("</s>"):
        caption = caption[:-4].strip()
    caption = " ".join(caption.replace("_", " ").split())
    return caption


def display_label(dataset: str, label: str) -> str:
    if dataset in {"food101", "ucf101"}:
        return label.replace("_", " ")
    return label


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
