# ============================================================
# CSE 5526 - Programming Assignment 2
# Voice Activity Detection in Noise using Deep Learning
# ============================================================

# --- Imports ---
from experiments import run_experiment
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*AudioDecoder.*", category=UserWarning)

# ============================================================
# run.ipynb
# ============================================================

results = []

DEFAULT_HIDDEN = 128
DEFAULT_LAYERS = 2
DEFAULT_SEQ = 50
DEFAULT_MODEL = "lstm"

# ------------------------------------------------------------
# 1️⃣ Test SEQUENCE LENGTH independently
# ------------------------------------------------------------
for seq_len in [10, 25, 50, 100]:
    print(f"\n🔹 Testing SEQ_LEN = {seq_len}")
    start = time.time()
    acc, prec, rec = run_experiment(
        DEFAULT_MODEL, n_features=40, hidden_size=DEFAULT_HIDDEN,
        layers=DEFAULT_LAYERS, seq_len=seq_len
    )
    end = time.time()

    results.append({
        "Factor": "SeqLen",
        "Value": seq_len,
        "Model": DEFAULT_MODEL.upper(),
        "HiddenUnit": DEFAULT_HIDDEN,
        "Layer": DEFAULT_LAYERS,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Time (s)": round(end - start, 1)
    })


# ------------------------------------------------------------
# 2️⃣ Test NUMBER OF LAYERS independently
# ------------------------------------------------------------
for num_layer in [1, 2, 3]:
    print(f"\n🔸 Testing NUM_LAYERS = {num_layer}")
    start = time.time()
    acc, prec, rec = run_experiment(
        DEFAULT_MODEL, n_features=40, hidden_size=DEFAULT_HIDDEN,
        layers=num_layer, seq_len=DEFAULT_SEQ
    )
    end = time.time()

    results.append({
        "Factor": "Layers",
        "Value": num_layer,
        "Model": DEFAULT_MODEL.upper(),
        "HiddenUnit": DEFAULT_HIDDEN,
        "Layer": num_layer,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Time (s)": round(end - start, 1)
    })


# ------------------------------------------------------------
# 3️⃣ Test HIDDEN UNITS independently
# ------------------------------------------------------------
for hidden_unit in [64, 128, 256]:
    print(f"\n🔹 Testing HIDDEN_UNIT = {hidden_unit}")
    start = time.time()
    acc, prec, rec = run_experiment(
        DEFAULT_MODEL, n_features=40, hidden_size=hidden_unit,
        layers=DEFAULT_LAYERS, seq_len=DEFAULT_SEQ
    )
    end = time.time()

    results.append({
        "Factor": "HiddenUnit",
        "Value": hidden_unit,
        "Model": DEFAULT_MODEL.upper(),
        "HiddenUnit": hidden_unit,
        "Layer": DEFAULT_LAYERS,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Time (s)": round(end - start, 1)
    })


# ------------------------------------------------------------
# 4️⃣ Test MODEL TYPES independently
# ------------------------------------------------------------
for model_name in ["lstm", "bilstm", "cnnlstm"]:
    print(f"\n🔸 Testing MODEL = {model_name.upper()}")
    start = time.time()
    acc, prec, rec = run_experiment(
        model_name, n_features=40, hidden_size=DEFAULT_HIDDEN,
        layers=DEFAULT_LAYERS, seq_len=DEFAULT_SEQ
    )
    end = time.time()

    results.append({
        "Factor": "ModelType",
        "Value": model_name.upper(),
        "Model": model_name.upper(),
        "HiddenUnit": DEFAULT_HIDDEN,
        "Layer": DEFAULT_LAYERS,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Time (s)": round(end - start, 1)
    })


# ------------------------------------------------------------
# Display results as a table
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("vad_results.csv", index=False)
print("Saved to vad_results.csv")