import torch
from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image


def load_model(model_name="google/vit-base-patch16-224"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model


def preprocess_image(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs.pixel_values


def forward_manual(pixel_values, model):
    # 1) exact embeddings
    hidden_states = model.vit.embeddings(pixel_values)  # (1, 1+N, D)

    # 2) first layer fully unpacked
    layer0 = model.vit.encoder.layer[0]

    self_attention_outputs = layer0.attention(
        # in ViT, layernorm is applied before self-attention
        layer0.layernorm_before(hidden_states),
        None,
        output_attentions=False,
    )
    attention_output = self_attention_outputs[0]

    # first residual connection
    hidden_states = attention_output + hidden_states

    # in ViT, layernorm is also applied after self-attention
    layer_output = layer0.layernorm_after(hidden_states)
    layer_output = layer0.intermediate(layer_output)

    # second residual connection is done here
    layer_output = layer0.output(layer_output, hidden_states)

    hidden_states = layer_output

    # 3) remaining layers via single‐tensor indexing
    for layer in model.vit.encoder.layer[1:]:
        hidden_states = layer(hidden_states)[0]

    # 4) final norm, pool, classify
    seq_out = model.vit.layernorm(hidden_states)
    pooled = seq_out[:, 0]
    return model.classifier(pooled)


if __name__ == "__main__":
    file_name = "images/Celular.jpg"
    processor, model = load_model()
    pixels = preprocess_image(file_name, processor)

    with torch.no_grad():
        # Manual forward
        manual_logits = forward_manual(pixels, model)

        # Hugging Face’s forward
        hf_logits = model(pixels).logits

    # Compute L2 difference
    diff = torch.norm(manual_logits - hf_logits).item()
    print(f"Diferença L2 entre manual vs HF logits: {diff:.6f}")

    val, ind = torch.topk(manual_logits, 5)
    score = torch.softmax(manual_logits, -1)
    print(f"\nArquivo {file_name} analisado. Pode ser:")

    for i in ind[0]:
        label = model.config.id2label[i.item()]
        print(f"	{label} com probabilidade {score[0][i].item()*100}%")
