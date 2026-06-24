from src.config import DATA_DIR, SAVED_MODELS_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, SEED
from src.data.mnist_loader import get_mnist_datasets, make_loader
from src.models.cnn_model import SimpleCNN
from src.training.train_baseline import train_model
from src.evaluation.metrics import evaluate_model
from src.utils.save_load import save_model
from src.utils.seed import set_seed
import json

def main():
    set_seed(SEED)
    train_dataset, test_dataset = get_mnist_datasets(DATA_DIR)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = make_loader(test_dataset, BATCH_SIZE, shuffle=False)
    model = SimpleCNN()
    model, runtime = train_model(model, train_loader, DEVICE, epochs=EPOCHS, lr=LEARNING_RATE)
    metrics = evaluate_model(model, test_loader, DEVICE)
    save_model(model, SAVED_MODELS_DIR / "baseline_model.pth")
    results = {"runtime": runtime, **metrics}
    print(results)

    # SAVE RESULTS
    results_path = SAVED_MODELS_DIR.parent / "results" / "baseline_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
