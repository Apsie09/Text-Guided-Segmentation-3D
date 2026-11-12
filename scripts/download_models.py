"""Pre-download the pretrained checkpoints used in the project into the HF cache.

Only config/tokenizer files and one copy of the weights are fetched
(safetensors preferred), skipping duplicate TF/Flax/PyTorch-bin weights.
"""
from huggingface_hub import list_repo_files, snapshot_download

MODELS = [
    "openai/clip-vit-base-patch16",
    "CIDAS/clipseg-rd64-refined",
]

META = ["*.json", "*.txt", "*.model"]


def weight_patterns(repo_id: str) -> list[str]:
    files = list_repo_files(repo_id)
    if any(f.endswith(".safetensors") for f in files):
        return ["*.safetensors"]
    return ["pytorch_model.bin"]


def main() -> None:
    for repo_id in MODELS:
        path = snapshot_download(repo_id, allow_patterns=META + weight_patterns(repo_id))
        print(f"{repo_id} -> {path}")


if __name__ == "__main__":
    main()
