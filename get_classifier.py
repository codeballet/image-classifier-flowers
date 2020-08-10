from torch import nn


# Build a new classifier
def get_classifier(units):
    classifier = nn.Sequential(nn.Linear(25088, 2048),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(2048, units),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(units, units),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(units, 102),
                               nn.LogSoftmax(dim=1))
    
    # Define loss function
    criterion = nn.NLLLoss()
    
    return criterion, classifier