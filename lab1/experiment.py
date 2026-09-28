import numpy as np
import torch
import torch.nn as nn
from data import load_and_preprocess_iris
from model import initialize_parameters, forward, compute_loss, backward

def run_pytorch_reference(X_train, y_train, params):
    model = nn.Sequential(
        nn.Linear(4, 8, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(8, 3, dtype=torch.float64)
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"]))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"]))

    criterion = nn.CrossEntropyLoss(reduction="mean")

    t_X = torch.from_numpy(X_train)
    t_y = torch.from_numpy(y_train)

    logits = model(t_X)
    loss = criterion(logits, t_y)
    loss.backward()

    torch_grads = {
        "W1": model[0].weight.grad.numpy().T,
        "b1": model[0].bias.grad.numpy(),
        "W2": model[2].weight.grad.numpy().T,
        "b2": model[2].bias.grad.numpy(),
    }
    return loss.item(), torch_grads


def compute_numerical_gradient(X, y, params, param_name, idx, eps=1e-6):
    original_val = params[param_name][idx]

    # L+
    params[param_name][idx] = original_val + eps
    Z2_plus, _ = forward(X, params)
    loss_plus, _ = compute_loss(Z2_plus, y)

    # L-
    params[param_name][idx] = original_val - eps
    Z2_minus, _ = forward(X, params)
    loss_minus, _ = compute_loss(Z2_minus, y)

    params[param_name][idx] = original_val

    return (loss_plus - loss_minus) / (2.0 * eps)


def run_pipeline(intentional_error=False):
    print("=" * 70)
    mode_str = "ДОСЛІД З ПОМИЛКОЮ (БЕЗ ДІЛЕННЯ НА N)" if intentional_error else "ЕТАЛОННИЙ ЗАПУСК"
    print(f"РЕЖИМ: {mode_str}")
    print("=" * 70)

    X_train, y_train, _, _ = load_and_preprocess_iris()
    params = initialize_parameters()

    # NumPy прохід
    Z2, cache = forward(X_train, params)
    loss_np, probs = compute_loss(Z2, y_train)
    grads_np = backward(cache, params, probs, y_train, intentional_error=intentional_error)

    # PyTorch еталон
    loss_pt, grads_pt = run_pytorch_reference(X_train, y_train, params)

    print(f"\nNumPy Loss:   {loss_np:.15f}")
    print(f"PyTorch Loss: {loss_pt:.15f}")

    print("\n" + "-" * 70)
    print("ТАБЛИЦЯ 1: ЗВІРКА З PYTORCH")
    print("-" * 70)
    print(f"{'Величина':<15} | {'Макс. різниця NumPy / PyTorch':<32} | {'Пройдено'}")
    print("-" * 70)

    loss_diff = abs(loss_np - loss_pt)
    print(f"{'Втрата':<15} | {loss_diff:<32.5e} | {loss_diff <= 1e-12}")

    for p in ["W1", "b1", "W2", "b2"]:
        diff = np.max(np.abs(grads_np[p] - grads_pt[p]))
        passed = (diff <= 1e-12)
        print(f"{'Градієнт ' + p:<15} | {diff:<32.5e} | {passed}")

    print("\n" + "-" * 70)
    print("ТАБЛИЦЯ 2: ЧИСЕЛЬНЕ ДИФЕРЕНЦІЮВАННЯ")
    print("-" * 70)
    print(f"{'Параметр':<12} | {'backward()':<18} | {'Чисельна похідна':<18} | {'Абс. різниця':<14} | {'Пройдено'}")
    print("-" * 70)

    check_params = [
        ("W1", (0, 0), "W1[0,0]"),
        ("b1", 0, "b1[0]"),
        ("W2", (0, 0), "W2[0,0]"),
        ("b2", 0, "b2[0]"),
    ]

    for p_name, idx, label in check_params:
        g_manual = grads_np[p_name][idx]
        g_num = compute_numerical_gradient(X_train, y_train, params, p_name, idx, eps=1e-6)
        abs_diff = abs(g_manual - g_num)
        passed = (abs_diff <= 1e-7)
        print(f"{label:<12} | {g_manual:<18.8e} | {g_num:<18.8e} | {abs_diff:<14.5e} | {passed}")
    print("-" * 70)
