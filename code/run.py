from torchvision import datasets, transforms

transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.EMNIST(root='data',split='balanced', train=True,
                                   download=False, transform=transform)