import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms, datasets, models
from sklearn.model_selection import train_test_split
from utils import max_samples_per_class
import matplotlib.pyplot as plt
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    num_epochs = 30

    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images if they are not the same size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization parameters for pre-trained models
    ])


    genmodels = ["stable_diffusion_2_1", "realistic_vision_1_4", "kandinsky_2_2", "openjourney_v4"]

    for theme in ["fruits"]:
        numberOfSamples = 1000

        # Create datasets
        dataset_real = datasets.ImageFolder(root=f"real_datasets/{theme}", transform=transform)
        train_set_real, test_set_real = train_test_split(dataset_real, test_size=0.2, random_state=42)
        train_real_indexes = max_samples_per_class(train_set_real, numberOfSamples)
        train_set_real = Subset(train_set_real, train_real_indexes)
        subset_real = Subset(train_set_real, np.random.choice(len(train_set_real), len(train_set_real) // 2, replace=False))

        # Create dataloaders
        train_loader_real = DataLoader(train_set_real, batch_size=32, shuffle=True)
        test_loader_real = DataLoader(test_set_real, batch_size=32, shuffle=True)
        subset_loader_real = DataLoader(subset_real, batch_size=32, shuffle=True)

        # Create real model
        model_real = models.resnet34()
        model_real.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512,
                out_features=10 if theme == "animals" else 5
            ),
            torch.nn.Sigmoid()
        )

        optimizer_real = torch.optim.Adam(model_real.parameters(), lr=0.001)
        scheduler_real = lr_scheduler.StepLR(optimizer_real, step_size=7, gamma=0.1)
        model_real.to(device)

        # Cretae subset model
        model_subset = models.resnet34()
        model_subset.fc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512,
                out_features=10 if theme == "animals" else 5
            ),
            torch.nn.Sigmoid()
        )

        optimizer_subset = torch.optim.Adam(model_subset.parameters(), lr=0.001)
        scheduler_subset = lr_scheduler.StepLR(optimizer_subset, step_size=7, gamma=0.1)
        model_subset.to(device)


        # Real model train
        for epoch in range(num_epochs):
            # Training pass
            model_real.train()
            running_loss = 0.0
            for images, labels in train_loader_real:
                images, labels = images.to(device), labels.to(device)

                optimizer_real.zero_grad()
                outputs = model_real(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_real.step()

                running_loss += loss.item()

            scheduler_real.step()

            if epoch % 5 == 0:
                print(f"Training Epoch [{epoch}/{num_epochs}], Real, {theme}")


        # Subset model train
        for epoch in range(num_epochs):
            # Training pass
            model_subset.train()
            running_loss = 0.0
            for images, labels in subset_loader_real:
                images, labels = images.to(device), labels.to(device)

                optimizer_subset.zero_grad()
                outputs = model_subset(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_subset.step()

                running_loss += loss.item()

            scheduler_subset.step()

            if epoch % 5 == 0:
                print(f"Training Epoch [{epoch}/{num_epochs}], Subset real, {theme}")

        
        # Real model test
        model_real.eval()
        total = 0
        total_correct = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader_real):
                outputs = model_real(images.to(device))
                _, predicted = torch.max(outputs, 1)
                total += 1
                total_correct += dataset_real.classes[labels[0]] == dataset_real.classes[predicted[0]]

        # Log accuracy
        with open(f"./testlogs.txt", "a") as f:
            f.write(f"Theme: {theme}, Dataset: Real, Accuracy: {total_correct / total}\n")


        # Subset model test
        model_subset.eval()
        total = 0
        total_correct = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader_real):
                outputs = model_subset(images.to(device))
                _, predicted = torch.max(outputs, 1)
                total += 1
                total_correct += dataset_real.classes[labels[0]] == dataset_real.classes[predicted[0]]

        # Log accuracy
        with open(f"./testlogs.txt", "a") as f:
            f.write(f"Theme: {theme}, Dataset: Subset real, Accuracy: {total_correct / total}\n")

        for genmodel in genmodels:
            # Create datasets
            dataset = datasets.ImageFolder(root=f"generated_datasets/{theme}/{genmodel}/256", transform=transform)
            subset = Subset(dataset, np.random.choice(len(dataset), len(dataset) // 2, replace=False))
            mixed_dataset = ConcatDataset([subset, subset_real])

            # Create dataloaders
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            mixed_loader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)

            # Create artificial model
            model = models.resnet34()
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=512
,
                    out_features=10 if theme == "animals" else 5
                ),
                torch.nn.Sigmoid()
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            model.to(device)

            # Create mixed model
            model_mixed = models.resnet34()
            model_mixed.fc = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=512
,
                    out_features=10 if theme == "animals" else 5
                ),
                torch.nn.Sigmoid()
            )

            optimizer_mixed = torch.optim.Adam(model_mixed.parameters(), lr=0.001)
            scheduler_mixed = lr_scheduler.StepLR(optimizer_mixed, step_size=7, gamma=0.1)
            model_mixed.to(device)

            # Training loop

            # Artificial model
            for epoch in range(num_epochs):
                # Training pass
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                scheduler.step()

                if epoch % 5 == 0:
                    print(f"Training Epoch [{epoch}/{num_epochs}], Artificial, {genmodel}, {theme}")


            # Mixed model
            for epoch in range(num_epochs):
                # Training pass
                model_mixed.train()
                running_loss = 0.0
                for images, labels in mixed_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer_mixed.zero_grad()
                    outputs = model_mixed(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_mixed.step()

                    running_loss += loss.item()

                scheduler_mixed.step()

                if epoch % 5 == 0:
                    print(f"Training Epoch [{epoch}/{num_epochs}], Mixed, {genmodel}, {theme}")


            # Test performance

            # Artificial model
            model.eval()
            total = 0
            total_correct = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader_real):
                    outputs = model(images.to(device))
                    _, predicted = torch.max(outputs, 1)
                    total += 1
                    total_correct += dataset_real.classes[labels[0]] == dataset_real.classes[predicted[0]]

            # Log accuracy
            with open(f"./testlogs.txt", "a") as f:
                f.write(f"Theme: {theme}, Model: {genmodel}, Dataset: Artificial, Accuracy: {total_correct / total}\n")



            # Mixed model
            model_mixed.eval()
            total = 0
            total_correct = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader_real):
                    outputs = model_mixed(images.to(device))
                    _, predicted = torch.max(outputs, 1)
                    total += 1
                    total_correct += dataset_real.classes[labels[0]] == dataset_real.classes[predicted[0]]

            # Log accuracy
            with open(f"./testlogs.txt", "a") as f:
                f.write(f"Theme: {theme}, Model: {genmodel}, Dataset: Mixed, Accuracy: {total_correct / total}\n")

    print("\n\n")

if __name__ == "__main__":
    main()