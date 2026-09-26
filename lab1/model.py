import numpy as np

def initialize_parameters():
    rng = np.random.default_rng(0)
    # W1: He ініціалізація (середнє 0, std = sqrt(2 / 4))
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4.0), size=(4, 8)).astype(np.float64)
    b1 = np.zeros(8, dtype=np.float64)

    # W2: Xavier ініціалізація (середнє 0, std = sqrt(2 / (8 + 3)))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8.0 + 3.0)), size=(8, 3)).astype(np.float64)
    b2 = np.zeros(3, dtype=np.float64)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(X, params):
    # Вхідний шар -> Прихований шар
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(0.0, Z1)  # ReLU

    # Прихований шар -> Вихідні логіти
    Z2 = A1 @ params["W2"] + params["b2"]

    # Збереження проміжних значень, критично необхідних для зворотного проходу
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2
    }
    return Z2, cache


def compute_loss(Z2, y):
    # Чисельно стабільний log-softmax зі зсувом на max(Z2)
    N = Z2.shape[0]
    shift_z = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    log_probs = shift_z - np.log(np.sum(exp_z, axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(N), y])
    return loss, probs


def backward(cache, params, probs, y, intentional_error=False):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    N = X.shape[0]

    # Градієнт крос-ентропії за вихідними логітами
    dZ2 = probs.copy()
    dZ2[np.arange(N), y] -= 1.0

    if not intentional_error:
        dZ2 /= N  # Усереднення за кількістю спостережень

    # Градієнти другого шару (використовують A1)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # Прохід крізь ReLU (використовує Z1 для перевірки умови Z1 > 0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0.0).astype(np.float64)

    # Градієнти першого шару (використовують вхідні дані X)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}