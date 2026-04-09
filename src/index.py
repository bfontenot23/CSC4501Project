import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CRC8_POLY = 0x07
CRC_BITS = 8


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        try:
            raw = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def prompt_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    options_text = ", ".join([f"{key}={label}" for key, label in choices.items()])
    while True:
        try:
            raw = input(f"{prompt} ({options_text}) [default={default}]: ").strip().lower()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"Please choose one of: {', '.join(choices.keys())}")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def crc8_bits(payload_bits: np.ndarray) -> np.ndarray:
    payload_int = _bits_to_int(payload_bits)
    padded = payload_int << CRC_BITS
    degree = payload_bits.size + CRC_BITS - 1
    for bit_pos in range(degree, CRC_BITS - 1, -1):
        if (padded >> bit_pos) & 1:
            padded ^= CRC8_POLY << (bit_pos - CRC_BITS)
    remainder = padded & ((1 << CRC_BITS) - 1)
    return _int_to_bits(remainder, CRC_BITS)


def make_codeword(payload_bits: np.ndarray) -> np.ndarray:
    return np.concatenate([payload_bits, crc8_bits(payload_bits)])


def crc_passes(codeword: np.ndarray) -> bool:
    payload = codeword[:-CRC_BITS]
    claimed_crc = codeword[-CRC_BITS:]
    expected_crc = crc8_bits(payload)
    return bool(np.array_equal(claimed_crc, expected_crc))


def flip_random_bits(bits: np.ndarray, rng: np.random.Generator, min_flips: int = 1, max_flips: int = 3) -> np.ndarray:
    corrupted = bits.copy()
    flips = int(rng.integers(min_flips, max_flips + 1))
    idx = rng.choice(corrupted.size, size=flips, replace=False)
    corrupted[idx] ^= 1
    return corrupted


def generate_dataset(
    samples: int,
    payload_len: int,
    corruption_rate: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((samples, payload_len + CRC_BITS), dtype=np.float32)
    y = np.zeros(samples, dtype=np.float32)

    for i in range(samples):
        payload = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
        codeword = make_codeword(payload)
        if rng.random() < corruption_rate:
            codeword = flip_random_bits(codeword, rng)

        x[i] = codeword.astype(np.float32)
        y[i] = 1.0 if crc_passes(codeword) else 0.0

    return x, y


def split_train_test(
    x: np.ndarray, y: np.ndarray, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = int(x.shape[0] * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


class CRCNet(nn.Module):
    def __init__(self, input_bits: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_bits, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, y_true.size)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return Metrics(accuracy, precision, recall, f1, tp, tn, fp, fn)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    seed: int,
    device: str,
    model: CRCNet | None = None,
) -> CRCNet:
    torch.manual_seed(seed)
    random.seed(seed)

    if model is None:
        model = CRCNet(input_bits=x_train.shape[1], hidden=hidden).to(device)
    else:
        model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)

        epoch_loss = running_loss / max(1, len(train_ds))
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"epoch={epoch:03d} loss={epoch_loss:.4f}")

    training_seconds = time.perf_counter() - start_time
    print(f"training_seconds={training_seconds:.2f}")
    return model


def predict_model(model: CRCNet, x: np.ndarray, device: str) -> np.ndarray:
    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x).to(device)
        probs = torch.sigmoid(model.forward(xb)).cpu().numpy()
    prediction_seconds = time.perf_counter() - start_time
    samples_per_second = x.shape[0] / max(prediction_seconds, 1e-9)
    print(f"inference_seconds={prediction_seconds:.4f} inference_samples_per_second={samples_per_second:.1f}")
    return (probs >= 0.5).astype(np.int64)


def evaluate_crc_baseline(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape[0], dtype=np.int64)
    x_uint8 = x.astype(np.uint8)
    for i in range(x_uint8.shape[0]):
        out[i] = 1 if crc_passes(x_uint8[i]) else 0
    return out


def save_dataset(path: str, x: np.ndarray, y: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not path.lower().endswith(".csv"):
        raise ValueError("Dataset files must use .csv extension.")
    headers = [f"bit_{i}" for i in range(x.shape[1])] + ["label"]
    data = np.column_stack([x.astype(np.uint8), y.astype(np.uint8)])
    np.savetxt(path, data, delimiter=",", fmt="%d", header=",".join(headers), comments="")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not path.lower().endswith(".csv"):
        raise ValueError("Dataset files must use .csv extension.")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    return x, y


def save_bitstreams(path: str, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not path.lower().endswith(".csv"):
        raise ValueError("Bitstream files must use .csv extension.")
    headers = [f"bit_{i}" for i in range(x.shape[1])]
    np.savetxt(path, x.astype(np.uint8), delimiter=",", fmt="%d", header=",".join(headers), comments="")


def load_bitstreams(path: str) -> np.ndarray:
    if not path.lower().endswith(".csv"):
        raise ValueError("Bitstream files must use .csv extension.")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data.astype(np.float32)


def save_test_results(path: str, x: np.ndarray, neural_net_pred: np.ndarray, actual_crc_label: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = [f"bit_{i}" for i in range(x.shape[1])] + [
        "neural_network_predicted_pass_or_fail",
        "actual_crc_pass_or_fail",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(x.shape[0]):
            bits = x[i].astype(np.uint8).tolist()
            nn_label = "pass" if int(neural_net_pred[i]) == 1 else "fail"
            actual_label = "pass" if int(actual_crc_label[i]) == 1 else "fail"
            writer.writerow(bits + [nn_label, actual_label])


def print_dataset_preview(name: str, x: np.ndarray, y: np.ndarray, rows: int) -> None:
    rows_to_show = min(rows, x.shape[0])
    print(f"\n{name} preview (showing {rows_to_show}/{x.shape[0]})")
    for i in range(rows_to_show):
        bits = "".join(str(int(bit)) for bit in x[i].astype(np.uint8))
        print(f"{name}[{i}] bits={bits} label={int(y[i])}")


def save_model(path: str, model: CRCNet, input_bits: int, hidden: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_bits": input_bits,
            "hidden": hidden,
        },
        path,
    )


def load_model(path: str, device: str) -> Tuple[CRCNet, int, int]:
    checkpoint = torch.load(path, map_location=device)
    input_bits = int(checkpoint["input_bits"])
    hidden = int(checkpoint["hidden"])
    model = CRCNet(input_bits=input_bits, hidden=hidden).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, input_bits, hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural-network CRC error detection experiment (V1)")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--payload-len", type=int, default=8)
    parser.add_argument("--corruption-rate", type=float, default=0.5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-checkpoint", type=str, default="models/crcnet.pt")
    parser.add_argument("--training-export", type=str, default="data/training_data.csv")
    parser.add_argument("--test-export", type=str, default="data/test_data.csv")
    parser.add_argument("--test-results-export", type=str, default="data/test_data_comparison.csv")
    parser.add_argument("--show-train-rows", type=int, default=0)
    parser.add_argument("--show-test-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    print(f"device={resolved_device}")

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    print(f"seed={seed} (pass --seed {seed} to reproduce this run)")

    train_samples = int(args.samples * (1 - args.test_ratio))
    test_samples = args.samples - train_samples
    if train_samples <= 0 or test_samples <= 0:
        raise ValueError("Choose --samples and --test-ratio so both train and test sizes are positive.")

    # Regenerate training data every run
    x_train, y_train = generate_dataset(
        samples=train_samples,
        payload_len=args.payload_len,
        corruption_rate=args.corruption_rate,
        seed=seed,
    )

    expected_bits = args.payload_len + CRC_BITS
    x_test: np.ndarray
    y_test: np.ndarray
    use_existing_test_data = False
    if os.path.exists(args.test_export):
        use_existing_test_data = prompt_yes_no(
            f"Existing test data found at {args.test_export}. Use this test data?",
            default=True,
        )

    if use_existing_test_data:
        x_test = load_bitstreams(args.test_export)
        if x_test.shape[1] != expected_bits:
            print(
                f"existing test data has bits_per_sample={x_test.shape[1]}, expected={expected_bits}; generating new test data"
            )
            x_test, y_test = generate_dataset(
                samples=test_samples,
                payload_len=args.payload_len,
                corruption_rate=args.corruption_rate,
                seed=seed + 1,
            )
        else:
            y_test = evaluate_crc_baseline(x_test).astype(np.float32)
            print(f"loaded existing test data <- {args.test_export}")
    else:
        x_test, y_test = generate_dataset(
            samples=test_samples,
            payload_len=args.payload_len,
            corruption_rate=args.corruption_rate,
            seed=seed + 1,
        )
        print("generated new test data for this run")

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    if args.show_train_rows > 0:
        print_dataset_preview("train", x_train, y_train, rows=args.show_train_rows)
    if args.show_test_rows > 0:
        print_dataset_preview("test", x_test, y_test, rows=args.show_test_rows)

    print(
        f"dataset={x.shape[0]} bits_per_sample={x.shape[1]} "
        f"train={x_train.shape[0]} test={x_test.shape[0]} pass_rate={y.mean():.3f}"
    )

    model_loaded = False
    model: CRCNet | None = None
    continue_training_loaded_model = False

    if os.path.exists(args.model_checkpoint):
        model_action = prompt_choice(
            f"Model checkpoint found at {args.model_checkpoint}. Choose action",
            {
                "u": "use existing model",
                "c": "continue training existing model",
                "f": "start over with a fresh model",
            },
            default="u",
        )

        if model_action in {"u", "c"}:
            model, model_input_bits, model_hidden = load_model(args.model_checkpoint, device=resolved_device)
            print(f"loaded model <- {args.model_checkpoint} (input_bits={model_input_bits}, hidden={model_hidden})")
            if model_input_bits != x_train.shape[1]:
                print(
                    f"checkpoint bits_per_sample={model_input_bits} does not match current={x_train.shape[1]}; starting fresh model"
                )
                model = None
            elif model_action == "u":
                model_loaded = True
            else:
                continue_training_loaded_model = True


    if not model_loaded:
        model = train_model(
            x_train=x_train,
            y_train=y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=args.hidden,
            seed=seed,
            device=resolved_device,
            model=model if continue_training_loaded_model else None,
        )

        if continue_training_loaded_model:
            print("continued training from existing model checkpoint")
        else:
            print("trained a fresh model")

        if prompt_yes_no("Do you want to export the model checkpoint?", default=True):
            save_model(args.model_checkpoint, model, input_bits=x_train.shape[1], hidden=args.hidden)
            print(f"saved model -> {args.model_checkpoint}")
        else:
            print("model export skipped")

        if prompt_yes_no("Do you want to export the training data used for this run?", default=False):
            save_dataset(args.training_export, x_train, y_train)
            print(f"saved training data export -> {args.training_export}")
        else:
            print("training data export skipped")

    if model is None:
        raise RuntimeError("Model initialization failed unexpectedly.")

    nn_pred = predict_model(model, x_test, device=resolved_device)
    nn_metrics = compute_metrics(y_test, nn_pred)

    crc_pred = evaluate_crc_baseline(x_test)
    crc_metrics = compute_metrics(y_test, crc_pred)

    print("\nNeural network metrics")
    print(
        f"accuracy={nn_metrics.accuracy:.4f} precision={nn_metrics.precision:.4f} "
        f"recall={nn_metrics.recall:.4f} f1_score={nn_metrics.f1:.4f}"
    )
    print(
        "confusion_matrix="
        f"true_positives={nn_metrics.tp} true_negatives={nn_metrics.tn} "
        f"false_positives={nn_metrics.fp} false_negatives={nn_metrics.fn}"
    )

    print("\nCRC baseline metrics")
    print(
        f"accuracy={crc_metrics.accuracy:.4f} precision={crc_metrics.precision:.4f} "
        f"recall={crc_metrics.recall:.4f} f1_score={crc_metrics.f1:.4f}"
    )
    print(
        "confusion_matrix="
        f"true_positives={crc_metrics.tp} true_negatives={crc_metrics.tn} "
        f"false_positives={crc_metrics.fp} false_negatives={crc_metrics.fn}"
    )

    print(
        "\ncomparison="
        f"accuracy_difference_vs_crc={nn_metrics.accuracy - crc_metrics.accuracy:.4f} "
        f"f1_difference_vs_crc={nn_metrics.f1 - crc_metrics.f1:.4f}"
    )

    if prompt_yes_no("Do you want to export test_data.csv and test_data_comparison.csv?", default=False):
        save_bitstreams(args.test_export, x_test)
        save_test_results(args.test_results_export, x_test, nn_pred, y_test)
        print(f"saved test data export -> {args.test_export}")
        print(f"saved test results export -> {args.test_results_export}")


if __name__ == "__main__":
    main()

