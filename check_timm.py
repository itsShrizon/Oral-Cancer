import timm
print("Available EfficientNet V2 models:")
for m in timm.list_models():
    if 'efficientnetv2' in m:
        print(m)
