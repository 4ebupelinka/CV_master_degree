import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Домашние задание №1
    Выполнил Тараканов Борис ЗФИмд-01-25 (1032259220)
    ### Вариант:
    **Датасет:** cifar10

    **Архитектура:** 3072->512->10

    **Активации:** ReLU, GELU

    **Лоссы:** CrossEntropy

    **Что перебрать:** lr {0.1, 0.01, 0.001}
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import random_split, DataLoader

    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import os

    torch.manual_seed(42)
    np.random.seed(42)

    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    return (
        BATCH_SIZE,
        DEVICE,
        DataLoader,
        NUM_EPOCHS,
        nn,
        optim,
        plt,
        random_split,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _(BATCH_SIZE, DataLoader, random_split, torchvision, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


@app.cell
def _(nn):
    class SimpleNN(nn.Module):
        def __init__(self, activation='relu'):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(3072, 512)
            self.fc2 = nn.Linear(512, 10)
        
            # Выбираем функцию активации
            if activation.lower() == 'relu':
                self.act = nn.ReLU()
            elif activation.lower() == 'gelu':
                self.act = nn.GELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
    
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x

    return (SimpleNN,)


@app.cell
def _(
    DEVICE,
    NUM_EPOCHS,
    SimpleNN,
    nn,
    optim,
    torch,
    train_loader,
    val_loader,
):
    def train_model(activation, lr):
        """Обучает модель с заданной активацией и learning rate.
           Возвращает словарь с историей потерь и лучшим val_loss."""
        model = SimpleNN(activation).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
    
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
            
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_train_loss += loss.item() * images.size(0)
        
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
        
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * images.size(0)
        
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
        
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
        
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train loss: {epoch_train_loss:.4f} | Val loss: {epoch_val_loss:.4f}")
    
        return {
            'activation': activation,
            'lr': lr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_val_loss': val_losses[-1]
        }

    return (train_model,)


@app.cell
def _(train_model):
    activations = ['relu', 'gelu']
    learning_rates = [0.1, 0.01, 0.001]

    results = []

    for act in activations:
        for lr in learning_rates:
            print(f"\n--- Запуск: activation={act}, lr={lr} ---")
            history = train_model(act, lr)
            results.append(history)

    print("\nВсе запуски завершены!")
    return (results,)


@app.cell
def _(plt, results):
    best_config = min(results, key=lambda x: x['best_val_loss'])

    print("=" * 50)
    print("Лучшая конфигурация:")
    print(f"Активация: {best_config['activation']}")
    print(f"Learning rate: {best_config['lr']}")
    print(f"Лучший val_loss: {best_config['best_val_loss']:.4f}")
    print(f"Финальный val_loss: {best_config['final_val_loss']:.4f}")
    print("=" * 50)

    plt.figure(figsize=(12, 6))
    for res in results:
        label = f"{res['activation']}, lr={res['lr']}"
        plt.plot(res['val_losses'], label=label, linestyle='--', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Сравнение валидационных потерь для всех конфигураций')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nСводная таблица:")
    print("Activation\tLR\tBest Val Loss\tFinal Val Loss")
    for res in results:
        print(f"{res['activation']}\t\t{res['lr']}\t{res['best_val_loss']:.4f}\t\t{res['final_val_loss']:.4f}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
