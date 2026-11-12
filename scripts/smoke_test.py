"""Check that the frozen CLIP encoders run on the GPU and expose what the plan needs.

Verifies:
  * vision hidden states are [B, 1 + 14*14, 768] at 224x224 (CLS + patch grid),
  * patch tokens reshape to [B, 768, 14, 14],
  * the projected text embedding is [B, 512],
  * peak VRAM for a batch of encoder inference in fp16.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch16"
BATCH = 16


def as_tensor(out):
    # transformers may return a tensor or a ModelOutput depending on version.
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"

    model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    for p in model.parameters():
        p.requires_grad_(False)

    images = [Image.new("RGB", (640, 480), (i * 15, 100, 200)) for i in range(BATCH)]
    texts = [f"the red cup on the left number {i}" for i in range(BATCH)]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        vis = model.vision_model(
            pixel_values=inputs["pixel_values"].half(), output_hidden_states=True
        )
        txt = as_tensor(
            model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        )

    hs = vis.hidden_states  # tuple: embeddings + one per transformer layer
    last = hs[-1]
    print(f"pixel_values      : {tuple(inputs['pixel_values'].shape)}")
    print(f"hidden states     : {len(hs)} tensors, each {tuple(last.shape)}")
    assert last.shape == (BATCH, 197, 768), last.shape

    patches = last[:, 1:, :]  # drop CLS
    grid = patches.transpose(1, 2).reshape(BATCH, 768, 14, 14)
    print(f"patch grid        : {tuple(grid.shape)}")

    print(f"text embedding    : {tuple(txt.shape)}")
    assert txt.shape == (BATCH, 512), txt.shape

    n_params = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"CLIP params       : {n_params / 1e6:.1f}M (frozen, fp16)")
    print(f"peak VRAM (B={BATCH}) : {peak:.0f} MiB")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
