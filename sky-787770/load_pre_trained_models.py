import torchvision.models as models

def load_pretrained_models(img_width=224, img_height=224):
    vgg16_base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg19_base = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    inception_base = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    resnet50_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
  
    return vgg16_base, vgg19_base, inception_base, resnet50_base