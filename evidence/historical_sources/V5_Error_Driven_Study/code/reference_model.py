"""The selection model: a linear scorer trained with a listwise softmax objective.

For every question the four candidates form one comparison group.  The scorer

    s_i = w . x_i

turns the group into a distribution p_i = softmax(s)_i.  The training target puts
uniform mass on the candidates whose final answer is correct, so the gradient
pushes probability away from the wrong candidates and towards the right ones.
This matches the evaluation metric exactly -- one correct pick earns the point --
instead of optimising a separate per-candidate calibration objective.

Group labels for which every candidate is correct carry no preference signal, so
they contribute nothing useful; training focuses on the disagreement groups.
"""
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - training only, not needed at inference
    _HAVE_SCIPY = False


@dataclass
class Scaler:
    """Column standardisation fitted on training groups only."""
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        self.std[self.std < 1e-8] = 1.0

    def apply(self, X):
        out = (X.astype(np.float64) - self.mean) / self.std
        return np.clip(out, -20.0, 20.0)


def _loss_grad(w, X, T, reg, mask=None, gw=None, temperature=1.0):
    """X [G,K,d] float64 standardised, T [G,K] rows of the target simplex.

    ``mask`` marks the real candidates when a group was padded out to a common
    width; padded slots are given a score floor so the softmax ignores them, and
    their target mass is already zero. ``gw`` is an optional per-group weight, used
    to tell the objective which comparisons matter most.

    ``temperature`` sharpens (tau < 1) or flattens (tau > 1) the model's own
    distribution inside the loss. The gradient is always sum_i (p_i - T_i) x_i, so
    tau only changes which p is used: at tau -> 0 the gradient is driven purely by
    whichever candidate currently sits on top, which is the quantity the metric
    actually reads.
    """
    scores = X @ w
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    if temperature == 1.0:
        lse_scaled = None
        scores = scores - scores.max(axis=1, keepdims=True)   # numerical stability
        lse = np.log(np.exp(scores).sum(axis=1, keepdims=True))
        per_group = lse[:, 0] - (T * scores).sum(axis=1)
        p = np.exp(scores - lse)
    else:
        z = scores / temperature
        z = z - z.max(axis=1, keepdims=True)
        lse_z = np.log(np.exp(z).sum(axis=1, keepdims=True))
        # tau * logsumexp(s/tau) - sum_i T_i s_i
        per_group = temperature * lse_z[:, 0] + scores.max(axis=1) \
            - (T * scores).sum(axis=1)
        p = np.exp(z - lse_z)
    if gw is None:
        loss = float(per_group.mean())
        grad = np.einsum('gk,gkd->d', p - T, X) / X.shape[0]
    else:
        total = float(gw.sum())
        loss = float((per_group * gw).sum() / total)
        grad = np.einsum('gk,gkd->d', (p - T) * gw[:, None], X) / total
    loss += 0.5 * float(np.sum(reg * w * w))
    grad += reg * w
    return loss, grad


def _hinge_loss_grad(w, X, y, reg, mask=None, margin=1.0):
    """Squared-hinge pairwise ranking: each correct candidate should outrank each
    wrong one by at least ``margin``.

    The metric only cares about which candidate is top, so matching pairs directly
    is arguably closer to it than fitting a whole distribution. Squared rather than
    plain hinge so the objective stays smooth enough for L-BFGS.
    """
    scores = X @ w                                        # [G,K]
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
        real = np.asarray(mask, dtype=np.float64)
    else:
        real = np.ones_like(np.asarray(y, dtype=np.float64))
    positive = (np.asarray(y, dtype=np.float64) > 0) * real
    negative = (np.asarray(y, dtype=np.float64) <= 0) * real
    paired = positive[:, :, None] * negative[:, None, :]  # [G,K,K]
    slack = np.clip(margin - (scores[:, :, None] - scores[:, None, :]), 0.0, None)
    slack = np.where(paired > 0, slack, 0.0)
    n_pairs = max(float(paired.sum()), 1.0)
    loss = float((slack ** 2).sum() / n_pairs) + 0.5 * float(np.sum(reg * w * w))
    # d/d s_i = -2 sum_j slack(i,j);  d/d s_j = +2 sum_i slack(i,j)
    dscores = 2.0 * (slack.sum(axis=1) - slack.sum(axis=2)) / n_pairs
    grad = np.einsum('gk,gkd->d', dscores, X) + reg * w
    return loss, grad


def fit(X, y, reg, *, n_groups=None, max_iter=400, seed=0, mask=None,
        objective='softmax', margin=1.0, group_weights=None, temperature=1.0):
    """Fit on disagreement groups. X [N,K,d] raw, y [N,K] 0/1 correctness."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if n_groups is None:
        keep = np.ones(len(X), dtype=bool)
    else:
        keep = np.zeros(len(X), dtype=bool)
        keep[:n_groups] = True
    if y.sum(axis=1).min() == 0:  # drop groups with no correct candidate
        keep &= y.sum(axis=1) > 0
    Xg, yg = X[keep], y[keep]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)[keep]
    if group_weights is not None:
        group_weights = np.asarray(group_weights, dtype=np.float64)[keep]
    targets = yg / yg.sum(axis=1, keepdims=True)
    kernel = _loss_grad if objective == 'softmax' else _hinge_loss_grad
    extra = (temperature,) if objective == 'softmax' else (margin,)
    d = X.shape[-1]
    reg = np.asarray(reg, dtype=np.float64).reshape(-1)
    if reg.size == 1:
        reg = np.full(d, float(reg))
    rng = np.random.default_rng(seed)
    best = None
    for attempt in range(2):
        w0 = rng.normal(0.0, 1e-3, d) if attempt == 0 else np.zeros(d)
        # the hinge needs the 0/1 labels, the softmax their simplex rows
        emitted = targets if objective == 'softmax' else yg
        args = (Xg, emitted, reg, mask, group_weights)
        if not _HAVE_SCIPY:
            w = _adam(w0, Xg, targets, reg, max_iter * 20, seed, mask, temperature)
        else:
            res = minimize(kernel, w0, args=args + extra,
                           jac=True, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'maxfun': max_iter * 2})
            w = res.x
        value = kernel(w, Xg, emitted, reg, mask, group_weights, *extra)[0]
        if best is None or value < best[0]:
            best = (value, w)
    return best[1]


def _adam(w0, X, T, reg, steps, seed, mask=None, temperature=1.0):
    """Fallback optimiser when SciPy is unavailable. Ignores group weights."""
    rng = np.random.default_rng(seed)
    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    lr = 0.05
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        _, grad = _loss_grad(w, X, T, reg, mask, None, temperature)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        w -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
        lr *= 0.9995
    return w


def predict(w, X, mask=None):
    """Argmax-of-scores selection over the real candidates of each group."""
    scores = np.asarray(X, dtype=np.float64) @ w
    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
    return scores.argmax(axis=1).astype(int)


def score_all(w, X):
    return np.asarray(X, dtype=np.float64) @ w


# ------------------------------------------------------------------ neural part
class MLP:
    """One hidden layer trained on the same listwise softmax objective.

    The linear scorer stays available as a skip connection (``w_lin``), so the
    network starts from the linear solution and only has to learn what it cannot
    express; with the skip frozen it can never do worse than the linear model.
    Everything is stored as plain NumPy arrays, which is what makes it exportable
    to a bundle that scores new banks with NumPy alone.
    """

    def __init__(self, d, hidden, w_lin=None, seed=0, nl_slice=None):
        rng = np.random.default_rng(seed)
        self.has_lin = w_lin is not None
        self.w_lin = (np.zeros(d) if w_lin is None
                      else np.asarray(w_lin, dtype=np.float64).copy())
        # Only part of the input needs to see the hidden layer. Restricting it to
        # the hand-built block keeps the dense embedding linear, which is both far
        # cheaper to train and much harder to overfit on 2k training groups.
        self.nl_slice = nl_slice
        width = d if nl_slice is None else nl_slice.stop - nl_slice.start
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(max(width, 1)), (width, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = np.zeros(hidden)
        self.b2 = np.zeros(1)

    def parameter_names(self):
        return ('W1', 'b1', 'W2', 'b2') + (('w_lin',) if self.has_lin else ())

    def forward(self, X):
        """X [G,K,d] -> scores [G,K]."""
        Z = X if self.nl_slice is None else X[:, :, self.nl_slice]
        hidden = np.tanh(Z @ self.W1 + self.b1)
        linear = X @ self.w_lin if self.has_lin else 0.0
        return linear + hidden @ self.W2 + self.b2

    def score(self, X):
        return self.forward(np.asarray(X, dtype=np.float64))

    def choices(self, X):
        return self.score(X).argmax(axis=1).astype(int)


def _mlp_grads(net, X, T, reg, reg_lin):
    """Objective and gradients for one batch of groups. X [G,K,d], T [G,K]."""
    groups = X.shape[0]
    Z = X if net.nl_slice is None else X[:, :, net.nl_slice]
    hidden = np.tanh(Z @ net.W1 + net.b1)
    scores = (X @ net.w_lin if net.has_lin else 0.0) + hidden @ net.W2 + net.b2
    scores = scores - scores.max(axis=1, keepdims=True)
    lse = np.log(np.exp(scores).sum(axis=1, keepdims=True))
    loss = float((lse - (T * scores).sum(axis=1)).mean())
    p = np.exp(scores - lse)
    ds = (p - T) / groups                                  # d loss / d score
    grads = {'W2': np.einsum('gk,gkh->h', ds, hidden),
             'b2': np.asarray([float(ds.sum())])}
    dhidden = ds[:, :, None] * net.W2[None, None, :] * (1.0 - hidden ** 2)
    grads['W1'] = np.einsum('gkd,gkh->dh', Z, dhidden)
    grads['b1'] = dhidden.sum(axis=(0, 1))
    penalty = 0.5 * reg * float((net.W1 * net.W1).sum() + net.W2 @ net.W2)
    grads['W1'] = grads['W1'] + reg * net.W1
    grads['W2'] = grads['W2'] + reg * net.W2
    if net.has_lin:
        grads['w_lin'] = np.einsum('gk,gkd->d', ds, X)
        penalty += 0.5 * reg_lin * float(net.w_lin @ net.w_lin)
        grads['w_lin'] = grads['w_lin'] + reg_lin * net.w_lin
    return loss + penalty, grads


def fit_mlp(X, y, reg, hidden=64, epochs=60, batch=128, lr=0.03, reg_lin=1e-3,
            seed=0, w_lin=None, nl_slice=None):
    """Train the network on disagreement groups. X [N,K,d] raw-standardised."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    targets = y / y.sum(axis=1, keepdims=True)
    net = MLP(X.shape[-1], hidden, w_lin=w_lin, seed=seed, nl_slice=nl_slice)
    keys = net.parameter_names()
    moment = {k: np.zeros_like(getattr(net, k)) for k in keys}
    velocity = {k: np.zeros_like(getattr(net, k)) for k in keys}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed + 1)
    steps_per_epoch = max(1, len(X) // batch)
    total = epochs * steps_per_epoch
    for step in range(1, total + 1):
        idx = rng.choice(len(X), size=min(batch, len(X)), replace=False)
        _, grads = _mlp_grads(net, X[idx], targets[idx], reg, reg_lin)
        for key in keys:
            param = getattr(net, key)
            moment[key] = beta1 * moment[key] + (1 - beta1) * grads[key]
            velocity[key] = beta2 * velocity[key] + (1 - beta2) * grads[key] ** 2
            mhat = moment[key] / (1 - beta1 ** step)
            vhat = velocity[key] / (1 - beta2 ** step)
            setattr(net, key, param - lr * mhat / (np.sqrt(vhat) + eps))
    return net

