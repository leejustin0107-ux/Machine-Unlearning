import copy
import time
import pandas as pd
import streamlit as st

from src.config import DATA_DIR, SAVED_MODELS_DIR, RESULTS_DIR, BATCH_SIZE, DEVICE, SEED
from src.data.mnist_loader import get_mnist_datasets, make_loader
from src.data.forget_split import create_forget_retain_split
from src.models.cnn_model import SimpleCNN
from src.training.train_baseline import train_model
from src.training.retrain_model import full_retrain
from src.unlearning.reverse_learning import reverse_learning_unlearn
from src.unlearning.retain_finetune import retain_finetune_unlearn
from src.unlearning.random_label_unlearning import random_label_unlearn
from src.evaluation.metrics import evaluate_model
from src.evaluation.privacy_verification import confidence_on_loader
from src.utils.save_load import save_model, load_model
from src.utils.seed import set_seed

st.set_page_config(page_title="AI Forgetting Control Center", layout="wide")
st.title("AI Forgetting Control Center")
st.caption("Local dashboard prototype for selective forgetting using machine unlearning")

set_seed(SEED)

def format_percent(value):
    return f"{value * 100:.2f}%"

def format_seconds(value):
    return f"{value:.2f}s"

with st.sidebar:
    st.header("Experiment Controls")
    forget_digit = st.selectbox("Forget target digit", list(range(10)), index=7)
    forget_size = st.slider("Forget size", min_value=50, max_value=2000, value=500, step=50)
    method = st.selectbox(
        "Method",
        ["Reverse Learning", "Retain Fine-Tuning", "Random-Label Unlearning", "Full Retraining"],
    )
    epochs = st.slider("Epochs", min_value=1, max_value=10, value=3)
    learning_rate = st.number_input("Learning rate", value=0.001, format="%.5f")
    alpha = st.slider("Forgetting strength alpha", 0.1, 2.0, 0.5, 0.1)

train_dataset, test_dataset = get_mnist_datasets(DATA_DIR)
test_loader = make_loader(test_dataset, BATCH_SIZE, shuffle=False)
forget_set, retain_set = create_forget_retain_split(train_dataset, forget_digit=forget_digit, forget_size=forget_size, seed=SEED)
forget_loader = make_loader(forget_set, BATCH_SIZE, shuffle=True)
retain_loader = make_loader(retain_set, BATCH_SIZE, shuffle=True)

baseline_path = SAVED_MODELS_DIR / "baseline_model.pth"
history_path = RESULTS_DIR / "experiment_history.csv"

col1, col2, col3 = st.columns(3)
col1.metric("Dataset", "MNIST")
col2.metric("Forget samples", len(forget_set))
col3.metric("Retain samples", len(retain_set))

st.subheader("1. Baseline CNN")
col_a, col_b = st.columns(2)

if col_a.button("Train Baseline CNN"):
    with st.spinner("Training baseline CNN..."):
        model = SimpleCNN()
        train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True)
        model, runtime = train_model(model, train_loader, DEVICE, epochs=epochs, lr=learning_rate)
        save_model(model, baseline_path)
        metrics = evaluate_model(model, test_loader, DEVICE)
        st.success(f"Baseline trained. Accuracy: {format_percent(metrics['accuracy'])}, Runtime: {format_seconds(runtime)}")

if col_b.button("Check Baseline Model"):
    if baseline_path.exists():
        model = load_model(SimpleCNN(), baseline_path, DEVICE)
        metrics = evaluate_model(model, test_loader, DEVICE)
        st.success(f"Baseline exists. Test accuracy: {format_percent(metrics['accuracy'])}")
    else:
        st.warning("No baseline model found yet. Train it first.")

st.subheader("2. Run Experiment")
if st.button("Run Selected Method"):
    if not baseline_path.exists() and method != "Full Retraining":
        st.error("Please train the baseline model first.")
    else:
        with st.spinner(f"Running {method}..."):
            start = time.time()
            if method == "Full Retraining":
                result = full_retrain(retain_loader, test_loader, forget_loader, DEVICE, epochs=epochs, lr=learning_rate)
            else:
                base_model = load_model(SimpleCNN(), baseline_path, DEVICE)
                if method == "Reverse Learning":
                    result = reverse_learning_unlearn(base_model, forget_loader, retain_loader, test_loader, DEVICE, epochs=epochs, lr=learning_rate, alpha=alpha)
                elif method == "Retain Fine-Tuning":
                    result = retain_finetune_unlearn(base_model, retain_loader, forget_loader, test_loader, DEVICE, epochs=epochs, lr=learning_rate)
                else:
                    result = random_label_unlearn(base_model, forget_loader, retain_loader, test_loader, DEVICE, epochs=epochs, lr=learning_rate)

            confidence = confidence_on_loader(result["model"], forget_loader, DEVICE)
            model_filename = f"{method.lower().replace(' ', '_')}_model.pth"
            save_model(result["model"], SAVED_MODELS_DIR / model_filename)

            row = {
                "method": method,
                "forget_digit": forget_digit,
                "forget_size": forget_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "runtime_seconds": result["runtime"],
                "test_accuracy": result["test"]["accuracy"],
                "retain_accuracy": result["retain"]["accuracy"],
                "forget_accuracy": result["forget"]["accuracy"],
                "forget_confidence": confidence["avg_confidence"],
                "total_wall_time": time.time() - start,
            }

            if history_path.exists():
                df = pd.read_csv(history_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_csv(history_path, index=False)

            st.success("Experiment completed and saved to history.")
            st.dataframe(pd.DataFrame([row]))

st.subheader("3. Experiment History")
if history_path.exists():
    history = pd.read_csv(history_path)
    display_history = history.copy()

    for col in ["test_accuracy", "retain_accuracy", "forget_accuracy", "forget_confidence"]:
        if col in display_history.columns:
            display_history[col] = display_history[col].apply(format_percent)

    for col in ["runtime_seconds", "total_wall_time"]:
        if col in display_history.columns:
            display_history[col] = display_history[col].apply(format_seconds)

    st.dataframe(display_history, use_container_width=True)

    chart_data = history.copy()
    chart_data[["test_accuracy", "retain_accuracy", "forget_accuracy"]] = (
        chart_data[["test_accuracy", "retain_accuracy", "forget_accuracy"]] * 100
    )
    st.bar_chart(chart_data.set_index("method")[["test_accuracy", "retain_accuracy", "forget_accuracy"]])
else:
    st.info("No experiments recorded yet.")

st.subheader("4. Notes")
st.write(
    "This is a base template. Tune hyperparameters, improve visualizations, and validate the unlearning results before using it for the final viva."
)
