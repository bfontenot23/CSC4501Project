import numpy as np

from src.index import (
    compute_metrics,
    evaluate_crc_baseline,
    generate_dataset,
    load_dataset,
    load_model,
    predict_model,
    save_dataset,
    save_model,
    split_train_test,
    train_model,
)


def test_end_to_end_smoke() -> None:
    x, y = generate_dataset(samples=500, payload_len=16, corruption_rate=0.5, seed=7)
    x_train, y_train, x_test, y_test = split_train_test(x, y, test_ratio=0.2, seed=7)

    model = train_model(
        x_train=x_train,
        y_train=y_train,
        epochs=2,
        batch_size=64,
        lr=1e-3,
        hidden=32,
        seed=7,
        device="cpu",
    )

    nn_pred = predict_model(model, x_test, device="cpu")
    nn_metrics = compute_metrics(y_test, nn_pred)

    baseline = evaluate_crc_baseline(x_test)
    baseline_metrics = compute_metrics(y_test, baseline)

    assert baseline_metrics.accuracy == 1.0
    assert np.isfinite(nn_metrics.accuracy)


def test_dataset_and_model_round_trip(tmp_path) -> None:
    x, y = generate_dataset(samples=300, payload_len=16, corruption_rate=0.5, seed=3)
    x_train, y_train, x_test, y_test = split_train_test(x, y, test_ratio=0.2, seed=3)

    train_path = tmp_path / "train_set.csv"
    test_path = tmp_path / "test_set.csv"
    model_path = tmp_path / "model.pt"

    save_dataset(str(train_path), x_train, y_train)
    save_dataset(str(test_path), x_test, y_test)

    loaded_x_train, loaded_y_train = load_dataset(str(train_path))
    loaded_x_test, loaded_y_test = load_dataset(str(test_path))

    assert np.array_equal(loaded_x_train, x_train)
    assert np.array_equal(loaded_y_train, y_train)
    assert np.array_equal(loaded_x_test, x_test)
    assert np.array_equal(loaded_y_test, y_test)

    model = train_model(
        x_train=loaded_x_train,
        y_train=loaded_y_train,
        epochs=1,
        batch_size=64,
        lr=1e-3,
        hidden=16,
        seed=3,
        device="cpu",
    )
    save_model(str(model_path), model, input_bits=loaded_x_train.shape[1], hidden=16)

    loaded_model, input_bits, hidden = load_model(str(model_path), device="cpu")
    assert input_bits == loaded_x_train.shape[1]
    assert hidden == 16

    preds = predict_model(loaded_model, loaded_x_test, device="cpu")
    assert preds.shape[0] == loaded_x_test.shape[0]


def test_dataset_csv_round_trip(tmp_path) -> None:
    x, y = generate_dataset(samples=50, payload_len=8, corruption_rate=0.5, seed=11)
    csv_path = tmp_path / "dataset.csv"

    save_dataset(str(csv_path), x, y)
    loaded_x, loaded_y = load_dataset(str(csv_path))

    assert np.array_equal(loaded_x, x)
    assert np.array_equal(loaded_y, y)


