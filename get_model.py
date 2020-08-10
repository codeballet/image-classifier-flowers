from torchvision import models


# Generate a new model
def get_model(arch):
    # Get torchvision model from user input
    try:
        print('Downloading model from torchvision...')
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)

        elif arch == 'vgg19':
            model = models.vgg19(pretrained=True)
            
        else:
            print('No valid model specified.')
            
        # Turn off gradients for the model
        for param in model.parameters():
            param.requires_grad = False
        
    except:
        print('Failed to download model from torchvision.')

    return model