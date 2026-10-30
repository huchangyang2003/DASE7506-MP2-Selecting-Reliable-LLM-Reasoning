"""Two convex linear objectives; NumPy L-BFGS adapted from our stage3_logistic.py."""
import numpy as np
from features import build_features
from arithmetic_features import build_arithmetic_features

EMBEDDING_WIDTH = 1024


def build_inputs(bank, arrays):
    base, names = build_features(bank, arrays, "group")
    extra, extra_names = build_arithmetic_features(bank, "arithmetic")
    extra = extra - extra.mean(axis=1, keepdims=True)
    x = np.concatenate((base, extra), axis=2)
    if x.shape != (len(bank), 4, 1071) or not np.isfinite(x).all():
        raise ValueError("Unexpected linear feature contract")
    return x, names + ["centered_" + name for name in extra_names]


def fit_scaler(x):
    flat = x.reshape(-1, x.shape[-1])
    mean, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-8] = 1
    return mean, scale


def loss_gradient(theta, x, y, objective, penalties):
    flat = x.reshape(-1, x.shape[-1])
    w, b = theta[:-1], theta[-1]
    z = (flat @ w + (b if objective == "bce" else 0)).reshape(y.shape)
    if objective == "bce":
        value = np.mean(np.logaddexp(0, z) - y * z)
        residual = (np.exp(-np.logaddexp(0, -z)) - y) / y.size
        bias_gradient = float(residual.sum())
    elif objective == "uniform_ce":
        shifted = z - z.max(axis=1, keepdims=True)
        exponential = np.exp(shifted)
        target = y / y.sum(axis=1, keepdims=True)
        log_denominator = np.log(exponential.sum(axis=1, keepdims=True))
        value = np.mean(np.sum(target * (log_denominator - shifted), axis=1))
        residual = (exponential / exponential.sum(axis=1, keepdims=True) - target) / len(y)
        bias_gradient = 0.0
    else:
        raise ValueError("Unknown objective")
    value += 0.5 * np.dot(penalties * w, w)
    gradient = np.empty_like(theta)
    gradient[:-1] = flat.T @ residual.reshape(-1) + penalties * w
    gradient[-1] = bias_gradient
    return float(value), gradient


def fit(x, y, objective, embedding_l2, hand_l2, max_iterations=2000, tolerance=1e-6):
    if x.ndim != 3 or x.shape[:2] != y.shape or not np.all(y.sum(axis=1) >= 1):
        raise ValueError("Invalid grouped supervision")
    if min(embedding_l2, hand_l2) <= 0:
        raise ValueError("Strictly positive block penalties required")
    penalties = np.full(x.shape[-1], hand_l2)
    penalties[:min(EMBEDDING_WIDTH, len(penalties))] = embedding_l2
    theta = np.zeros(x.shape[-1] + 1)
    if objective == "bce":
        rate = float(np.clip(y.mean(), 1e-8, 1-1e-8))
        theta[-1] = np.log(rate / (1-rate))
    value, gradient = loss_gradient(theta, x, y, objective, penalties)
    memory, trace = [], []
    reason, evaluations = "max_iterations", 1
    for iteration in range(max_iterations + 1):
        gradient_inf = float(np.abs(gradient).max())
        trace.append({"iteration": iteration, "loss": value, "gradient_inf": gradient_inf})
        if gradient_inf <= tolerance:
            reason = "gradient_tolerance"
            break
        if iteration == max_iterations:
            break
        direction, alphas = gradient.copy(), []
        for step, change, rho in reversed(memory):
            alpha = rho * np.dot(step, direction)
            alphas.append(alpha)
            direction -= alpha * change
        if memory:
            step, change, _ = memory[-1]
            direction *= np.dot(step, change) / np.dot(change, change)
        for (step, change, rho), alpha in zip(memory, reversed(alphas)):
            direction += step * (alpha - rho * np.dot(change, direction))
        direction = -direction
        slope = float(np.dot(gradient, direction))
        if slope >= 0:
            direction, slope = -gradient, -float(np.dot(gradient, gradient))
            memory.clear()
        step_length = 1.0
        for _ in range(40):
            proposed = theta + step_length * direction
            new_value, new_gradient = loss_gradient(proposed, x, y, objective, penalties)
            evaluations += 1
            if np.isfinite(new_value) and np.isfinite(new_gradient).all() and new_value <= value + 1e-4 * step_length * slope:
                break
            step_length *= 0.5
        else:
            reason = "line_search_failed"
            break
        step, change = proposed-theta, new_gradient-gradient
        curvature = float(np.dot(step, change))
        if curvature > 1e-12:
            memory.append((step, change, 1/curvature))
            memory = memory[-10:]
        theta, value, gradient = proposed, new_value, new_gradient
    return theta, {"converged": reason == "gradient_tolerance", "stop_reason": reason,
                   "iterations": len(trace)-1, "final_loss": value,
                   "gradient_inf": float(np.abs(gradient).max()), "evaluations": evaluations,
                   "trace": trace}


def scores(x, model):
    return ((x-model["mean"])/model["scale"]) @ model["weights"] + float(model["bias"])
