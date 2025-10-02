import diffusers, transformers, torch
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())