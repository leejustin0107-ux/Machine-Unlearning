import copy
import time
import pandas as pd
import streamlit as st
import altair as alt
from pandas.errors import EmptyDataError, ParserError

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
from src.evaluation.prediction_examples import get_prediction_examples
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

HISTORY_COLUMNS = [
    "method",
    "forget_digit",
    "forget_size",
    "epochs",
    "learning_rate",
    "runtime_seconds",
    "test_accuracy",
    "retain_accuracy",
    "forget_accuracy",
    "forget_confidence",
    "total_wall_time",
]


def load_experiment_history(history_path):
    if not history_path.exists() or history_path.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        return pd.read_csv(history_path)
    except (EmptyDataError, ParserError):
        broken_path = history_path.with_name("experiment_history_broken.csv")
        history_path.rename(broken_path)

        st.warning(
            "Experiment history file was empty or corrupted, so it was reset. "
            "A backup was saved as experiment_history_broken.csv."
        )

        return pd.DataFrame(columns=HISTORY_COLUMNS)

def show_prediction_verification(examples, method_name):
    st.subheader("Prediction Verification on Forget Samples")
    st.caption(
        "This section compares how the baseline model and the selected method predict the same forget samples."
    )

    if not examples:
        st.info("No prediction examples available.")
        return

    cols = st.columns(3)

    for index, example in enumerate(examples):
        col = cols[index % 3]

        with col:
            st.image(
                example["image"],
                caption=f"True label: {example['true_label']}",
                width=100,
            )

            st.write(
                f"**Baseline:** {example['baseline_pred']} "
                f"({format_percent(example['baseline_conf'])})"
            )

            st.write(
                f"**{method_name}:** {example['unlearned_pred']} "
                f"({format_percent(example['unlearned_conf'])})"
            )
  
def clean_numeric_column(series):
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("s", "", regex=False)
        .str.replace("None", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def normalize_percentage_column(series):
    series = clean_numeric_column(series)

    # If values are decimals like 0.98, convert to 98%
    # If values are already like 98.8, keep them
    if series.dropna().max() <= 1:
        return series * 100

    return series


def show_experiment_charts(history):
    st.subheader("Experiment Charts")

    if history.empty:
        st.info("No experiment history available yet.")
        return

    chart_data = history.copy()

    numeric_cols = [
        "test_accuracy",
        "retain_accuracy",
        "forget_accuracy",
        "forget_confidence",
        "runtime_seconds",
        "epochs",
        "forget_size",
        "alpha",
    ]

    for col in numeric_cols:
        if col in chart_data.columns:
            chart_data[col] = clean_numeric_column(chart_data[col])

    for col in ["test_accuracy", "retain_accuracy", "forget_accuracy", "forget_confidence"]:
        if col in chart_data.columns:
            chart_data[col] = normalize_percentage_column(chart_data[col])

    chart_data["forget_effect"] = 100 - chart_data["forget_accuracy"]

    st.caption(
        "These charts compare accuracy preservation, forgetting effect, and runtime. "
        "Forget Effect = 100% - Forget Accuracy."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Method Comparison", "Runtime", "Reverse Learning Tuning"]
    )

    with tab1:
        st.markdown("#### Method Comparison")

        plot_data = chart_data.groupby("method", as_index=False).tail(1)

        metric_data = plot_data.melt(
            id_vars=["method"],
            value_vars=["test_accuracy", "retain_accuracy", "forget_effect"],
            var_name="metric",
            value_name="percentage",
        )

        metric_data["metric"] = metric_data["metric"].replace(
            {
                "test_accuracy": "Test Accuracy",
                "retain_accuracy": "Retain Accuracy",
                "forget_effect": "Forget Effect",
            }
        )

        metric_data = metric_data.dropna(subset=["percentage"])

        chart = (
            alt.Chart(metric_data)
            .mark_bar()
            .encode(
                x=alt.X("method:N", title="Method"),
                xOffset=alt.XOffset("metric:N"),
                y=alt.Y(
                    "percentage:Q",
                    title="Percentage (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=[
                    alt.Tooltip("method:N", title="Method"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("percentage:Q", title="Value", format=".2f"),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)

        st.info(
            "Forget Effect = 100% - Forget Accuracy. "
            "Higher forget effect means stronger forgetting."
        )

    with tab2:
        st.markdown("#### Runtime Comparison")

        runtime_data = chart_data.groupby("method", as_index=False).tail(1)
        runtime_data = runtime_data.dropna(subset=["runtime_seconds"])

        runtime_chart = (
            alt.Chart(runtime_data)
            .mark_bar()
            .encode(
                y=alt.Y("method:N", title=None, sort="-x"),
                x=alt.X("runtime_seconds:Q", title="Runtime (seconds)"),
                tooltip=[
                    alt.Tooltip("method:N", title="Method"),
                    alt.Tooltip("runtime_seconds:Q", title="Runtime", format=".2f"),
                ],
            )
            .properties(height=250)
        )

        st.altair_chart(runtime_chart, use_container_width=True)

        st.caption(
            "Lower runtime is better. Retain Fine-Tuning and Full Retraining usually take longer because they train on a larger dataset."
        )

    with tab3:
        st.markdown("#### Reverse Learning Alpha Tuning")

        reverse_data = chart_data[chart_data["method"] == "Reverse Learning"].copy()

        if reverse_data.empty or "alpha" not in reverse_data.columns:
            st.info("No Reverse Learning tuning data available yet.")
            return

        reverse_data = reverse_data.dropna(subset=["alpha"])

        if reverse_data.empty:
            st.info("No Reverse Learning alpha values available yet.")
            return

        reverse_metric_data = reverse_data.melt(
            id_vars=["alpha"],
            value_vars=["test_accuracy", "retain_accuracy", "forget_accuracy"],
            var_name="metric",
            value_name="percentage",
        )

        reverse_metric_data["metric"] = reverse_metric_data["metric"].replace(
            {
                "test_accuracy": "Test Accuracy",
                "retain_accuracy": "Retain Accuracy",
                "forget_accuracy": "Forget Accuracy",
            }
        )

        alpha_chart = (
            alt.Chart(reverse_metric_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("alpha:Q", title="Forgetting Strength Alpha"),
                y=alt.Y(
                    "percentage:Q",
                    title="Accuracy (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=[
                    alt.Tooltip("alpha:Q", title="Alpha"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("percentage:Q", title="Value", format=".2f"),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(alpha_chart, use_container_width=True)

        st.caption(
            "Higher alpha usually increases forgetting, but may reduce test and retain accuracy."
        )

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

            prediction_examples = []

            if baseline_path.exists():
                baseline_model_for_examples = load_model(SimpleCNN(), baseline_path, DEVICE)
                prediction_examples = get_prediction_examples(
                    baseline_model_for_examples,
                    result["model"],
                    forget_loader,
                    DEVICE,
                    max_samples=6,
                )

            row = {
                "method": method,
                "forget_digit": forget_digit,
                "forget_size": forget_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "alpha": alpha if method == "Reverse Learning" else None,
                "runtime_seconds": result["runtime"],
                "test_accuracy": result["test"]["accuracy"],
                "retain_accuracy": result["retain"]["accuracy"],
                "forget_accuracy": result["forget"]["accuracy"],
                "forget_confidence": confidence["avg_confidence"],
                "total_wall_time": time.time() - start,
            }

            df = load_experiment_history(history_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(history_path, index=False)

            st.success("Experiment completed and saved to history.")
            display_row = row.copy()

            for col in ["test_accuracy", "retain_accuracy", "forget_accuracy", "forget_confidence"]:
                display_row[col] = f"{display_row[col] * 100:.2f}%"

            for col in ["runtime_seconds", "total_wall_time"]:
                display_row[col] = f"{display_row[col]:.2f}s"

            st.dataframe(pd.DataFrame([display_row]), use_container_width=True)

            show_prediction_verification(prediction_examples, method)

st.subheader("3. Experiment History")

history = load_experiment_history(history_path)

if history.empty:
    st.info("No experiments recorded yet.")
else:
    # Add delete controls
    st.markdown("#### Manage History")

    history_with_id = history.reset_index().rename(columns={"index": "run_id"})

    history_with_id["delete_label"] = history_with_id.apply(
        lambda row: (
            f"Run {row['run_id']} | {row['method']} | "
            f"digit {row['forget_digit']} | size {row['forget_size']} | "
            f"epochs {row['epochs']}"
        ),
        axis=1,
    )

    selected_runs = st.multiselect(
        "Select experiment rows to delete",
        history_with_id["delete_label"].tolist(),
    )

    col_delete, col_clear = st.columns(2)

    with col_delete:
        if st.button("Delete Selected Rows"):
            if selected_runs:
                selected_ids = history_with_id[
                    history_with_id["delete_label"].isin(selected_runs)
                ]["run_id"].tolist()

                updated_history = history.drop(index=selected_ids)
                updated_history.to_csv(history_path, index=False)

                st.success("Selected experiment rows deleted.")
                st.rerun()
            else:
                st.warning("Please select at least one row to delete.")

    with col_clear:
        if st.button("Clear All History"):
            history_path.unlink(missing_ok=True)
            st.success("All experiment history cleared.")
            st.rerun()

    # Display formatted history table
    display_history = history.copy()

    for col in [
        "test_accuracy",
        "retain_accuracy",
        "forget_accuracy",
        "forget_confidence",
        "forget_true_label_confidence",
    ]:
        if col in display_history.columns:
            display_history[col] = display_history[col].apply(format_percent)

    for col in ["runtime_seconds", "total_wall_time"]:
        if col in display_history.columns:
            display_history[col] = display_history[col].apply(format_seconds)

    st.dataframe(display_history, use_container_width=True)

    show_experiment_charts(history)

