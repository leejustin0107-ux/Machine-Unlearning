# Machine Unlearning Prototype Template

A GitHub-ready base template for a capstone prototype on **Selective Forgetting in Deep Learning Models Using Machine Unlearning**.

The prototype is designed as a **local Python/Streamlit dashboard** that demonstrates how selected MNIST training data can be forgotten from a CNN model and compared against full retraining.

## Prototype Goal

This project evaluates whether machine unlearning can:

1. Reduce the influence of selected forget data.
2. Preserve model performance on retained data.
3. Run faster than full retraining.
4. Provide visual evidence through metrics, graphs, and experiment history.

## Included Methods

This template includes starter files for:

- Full retraining benchmark
- Reverse learning / gradient ascent unlearning
- Retain-set fine-tuning
- Random-label unlearning as a third comparison method

You can replace the third method with a SISA-style partition-based method later if your supervisor prefers that direction.

## Folder Structure

```text
machine-unlearning-prototype-template/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── unlearning/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── saved_models/
├── results/
├── reports/
├── docs/
└── notebooks/
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open locally in your browser, usually at:

```text
http://localhost:8501
```

## Suggested Development Order

1. Confirm MNIST loads correctly.
2. Train the baseline CNN.
3. Create forget and retain sets.
4. Implement full retraining benchmark.
5. Implement unlearning methods.
6. Evaluate accuracy, loss, runtime, retain accuracy, and forget accuracy.
7. Add graphs and experiment history.
8. Polish the dashboard for viva demonstration.

## Important Notes

- The current version uses MNIST and a fixed CNN architecture for controlled evaluation.
- Custom dataset upload and custom model upload are future work.
- This template is a starting point. The group should tune hyperparameters, validate results, and improve the dashboard.
