import collections
import torch, torchvision

# Mac users
torch.set_default_device(torch.device("mps"))
# CUDA users
# torch.set_default_device(torch.device("cuda"))

torch.manual_seed(37)

weights = torchvision.models.ResNet50_Weights.DEFAULT
res = torchvision.models.resnet50(weights = weights)

def get_dataloaders(
    batch_size: int, train_proportion: float,
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Load the data in a form that the torch model can understand."""
    path_to_data = "defungi"
    # Set up preprocessing, allow for more functions
    transforms = torchvision.transforms.Compose([
        weights.transforms(), # preprocessing function
    ])
    # First, build the datasets.  Later, DataLoader will break into batches.
    # torchvision's ImageFolder is much like 
    #   Tensorflow's image_dataset_from_directory()
    full_dataset = torchvision.datasets.ImageFolder(
        "defungi",
        transform = transforms,
    )
    generator = torch.Generator(device="mps").manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
        full_dataset,
        [train_proportion, 1-train_proportion],
        generator,
    )
    # We have datasets, let's get dataloaders
    train = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    validation = torch.utils.data.DataLoader(validation_set, 
            batch_size = batch_size)
    return train, validation

train, validation = get_dataloaders(batch_size = 32, train_proportion = 0.8)

print(f"Number of train batches: {len(train)}")
print(f"Number of validation batches: {len(validation)}")

# [modelname].children shows all the layers
# print(res.children)

# Identity means just pass the data through to the next layer unchanged.
res.fc = torch.nn.Identity()

model = torch.nn.Sequential(collections.OrderedDict([
    ('resnet', res),
    ('final', torch.nn.LazyLinear(5)), # The size of this matches # of classes
    ('softmax', torch.nn.Softmax(dim=1)),
]))

lr = 0.0001
param_groups = [
    {'params': model.resnet.parameters(), 'requires_grad': False},
    {'params': model.final.parameters(), 'lr': lr},
]
optimizer = torch.optim.Adam(param_groups)
# Set the model to training mode.
model.train()

loss_fn = torch.nn.CrossEntropyLoss()

# The number in the next line is # of epochs
for i in range(3):
    print(f"========= EPOCH {i+1} =========")
    batch_losses = []
    batch_accuracies = []
    # Training Loop
    for image_batch, label_batch in train:
        preds = model(image_batch)
        # The loss object is *waaay* bigger than just a float.
        loss = loss_fn(preds, label_batch)
        # Set up the backpropogation step.
        loss.backward()
        # Actually carry out the training.
        optimizer.step()
        # Clear the results so we're ready for the next batch.
        optimizer.zero_grad()
        # The rest of the training loop is just printout.
        batch_losses.append(float(loss))
        cur_loss = sum(batch_losses)/len(batch_losses)
        this_accuracy = (int(sum(preds.argmax(1) == label_batch))/
                len(label_batch))
        batch_accuracies.append(this_accuracy)
        cur_accuracy = sum(batch_accuracies)/len(batch_accuracies)
        print("Train:", end="\t\t")
        print(f"Batch: {len(batch_losses)}", end="\t")
        print(f"Loss: {round(cur_loss, 4)}", end="\t")
        print(f"Accuracy: {round(cur_accuracy, 4)}", end="\r")
    print()
    batch_losses = []
    batch_accuracies = []
    # Validation Loop
    for image_batch, label_batch in validation:
        preds = model(image_batch)
        # The loss object is *waaay* bigger than just a float.
        loss = loss_fn(preds, label_batch)
        # The rest of the training loop is just printout.
        batch_losses.append(float(loss))
        cur_loss = sum(batch_losses)/len(batch_losses)
        this_accuracy = (int(sum(preds.argmax(1) == label_batch))/
                len(label_batch))
        batch_accuracies.append(this_accuracy)
        cur_accuracy = sum(batch_accuracies)/len(batch_accuracies)
        print("Validation:", end="\t")
        print(f"Batch: {len(batch_losses)}", end="\t")
        print(f"Loss: {round(cur_loss, 4)}", end="\t")
        print(f"Accuracy: {round(cur_accuracy, 4)}", end="\r")
    print()