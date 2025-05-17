import torch
from transformers import pipeline

from os import listdir
from os.path import isfile, join

imagePath = "./images"

modelName = "google/vit-base-patch16-224" ## "google/vit-base-patch16-224"

pipeline = pipeline(
    task="image-classification",
    model=modelName,
    torch_dtype=torch.float16,
    device=0
)

print("------------ Finish setup ------------")

files = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]

for i in files:
	result = pipeline(images=f"{imagePath}/{i}")
	
	print(f"\nArquivo {i} analisado. Pode ser:")

	for option in result:
		print(f"	{option["label"]} com probabilidade {option["score"]*100}%")