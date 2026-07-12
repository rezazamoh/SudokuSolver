import os
import csv
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from models.cnn import SudokuCNN


# ==========================
# Configuration
# ==========================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

WEIGHTS_DIR = "weights"

BATCH_SIZE = 256
NUM_WORKERS = 4

EPOCHS = 50

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================
# Accuracy Function
# ==========================

def calculate_accuracy(outputs, labels):

    predictions = outputs.argmax(dim=1)

    correct = (predictions == labels).sum().item()

    return correct / labels.size(0)


# ==========================
# Save Model
# ==========================

def save_checkpoint(model, optimizer, epoch, loss):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        },
        os.path.join(
            WEIGHTS_DIR,
            "best_model.pth"
        )
    )

def main():

    # بهینه‌سازی برای GPU
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    os.makedirs(
        WEIGHTS_DIR,
        exist_ok=True
    )

    log_file = "training_log.csv"

    with open(log_file, "w", newline="") as f:

        csv_writer = csv.writer(f)

        csv_writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "learning_rate"
        ])

    tb_writer = SummaryWriter("runs/sudoku_cnn")

    print("=" * 50)
    print("Sudoku Digit Classifier")
    print("=" * 50)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(torch.cuda.get_device_name(0))

    print("=" * 50)


    # ==========================
    # Data Augmentation
    # ==========================

    train_transform = transforms.Compose([

        transforms.Grayscale(),

        transforms.RandomRotation(8),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05)
        ),

        transforms.ToTensor()

    ])

    test_transform = transforms.Compose([

        transforms.Grayscale(),

        transforms.ToTensor()

    ])


    # ==========================
    # Dataset
    # ==========================

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )


    # ==========================
    # DataLoader
    # ==========================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(DEVICE.type == "cuda"),

        persistent_workers=NUM_WORKERS > 0

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(DEVICE.type == "cuda"),

        persistent_workers=NUM_WORKERS > 0

    )

    print("Train Images :", len(train_dataset))
    print("Test Images  :", len(test_dataset))

    # ==========================
    # Model
    # ==========================

    model = SudokuCNN(num_classes=10).to(DEVICE)


    # ==========================
    # Loss Function
    # ==========================

    criterion = nn.CrossEntropyLoss()


    # ==========================
    # Optimizer
    # ==========================

    optimizer = AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # ==========================
    # Learning Rate Scheduler
    # ==========================

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3,

        min_lr=1e-6

    )


    # ==========================
    # Mixed Precision
    # ==========================

    USE_AMP = DEVICE.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )


    # ==========================
    # Early Stopping
    # ==========================

    best_loss = float("inf")

    epochs_without_improvement = 0
    # ==========================
    # Training Loop
    # ==========================

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # --------------------------
        # Train
        # --------------------------

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        train_bar = tqdm(
            train_loader,
            desc="Training",
            leave=False
        )

        for images, labels in train_bar:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            acc = calculate_accuracy(outputs, labels)

            train_loss += loss.item()
            train_acc += acc

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc*100:.2f}%"
            )

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # --------------------------
        # Validation
        # --------------------------

        model.eval()

        val_loss = 0.0
        val_acc = 0.0

        val_bar = tqdm(
            test_loader,
            desc="Validation",
            leave=False
        )

        with torch.no_grad():

            for images, labels in val_bar:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=USE_AMP
                ):

                    outputs = model(images)

                    loss = criterion(outputs, labels)

                acc = calculate_accuracy(
                    outputs,
                    labels
                )

                val_loss += loss.item()
                val_acc += acc

        val_loss /= len(test_loader)
        val_acc /= len(test_loader)

        scheduler.step(val_loss)

        print(
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_acc*100:.2f}%"
        )

        print(
            f"Val Loss   : {val_loss:.4f} | "
            f"Val Acc   : {val_acc*100:.2f}%"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning Rate : {current_lr:.6f}")

        
        # ==========================
        # CSV Logging
        # ==========================

        with open(log_file, "a", newline="") as f:

            csv_writer = csv.writer(f)

            csv_writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr
            ])


        # ==========================
        # TensorBoard Logging
        # ==========================

        tb_writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch
        )

        tb_writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch
        )

        tb_writer.add_scalar(
            "Accuracy/Train",
            train_acc,
            epoch
        )

        tb_writer.add_scalar(
            "Accuracy/Validation",
            val_acc,
            epoch
        )

        tb_writer.add_scalar(
            "Learning Rate",
            current_lr,
            epoch
        )

        # --------------------------
        # Save Best Model
        # --------------------------

        if val_loss < best_loss:

            best_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss
            )

            print("✓ Best model saved.")

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement ({epochs_without_improvement}/{PATIENCE})"
            )

        # --------------------------
        # Early Stopping
        # --------------------------

        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

    tb_writer.close()

    print("\nTraining Finished!")

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()