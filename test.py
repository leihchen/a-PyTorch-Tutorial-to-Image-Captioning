import numpy as np
import torch
import clip
import torchvision

for m in clip.available_models():
# device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(m, device="cpu", jit=False)
    model.eval()
    print(type(model))
    print(type(torchvision.models.resnet101(pretrained=True)))
    for i,mc in enumerate(list(model.children())):
        print(i, mc)
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

