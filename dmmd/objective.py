import torch
import torch.distributed as dist


class _AllReduceSum(torch.autograd.Function):
    """Sum in forward AND backward; required for gradients through global means."""

    @staticmethod
    def forward(ctx, value):
        result = value.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    @staticmethod
    def backward(ctx, gradient):
        return _AllReduceSum.apply(gradient)


def linear_statistics(a, b):
    """Sufficient statistics, retaining all parameter-gradient paths."""
    return torch.cat((
        a.sum(dim=0), b.sum(dim=0),
        a.square().sum().reshape(1), b.square().sum().reshape(1),
        a.new_tensor([len(a), len(b)]),
    ))


def discrepancy_from_statistics(statistics, dimension):
    sa = statistics[:dimension]
    sb = statistics[dimension:2 * dimension]
    qa, qb, n, m = statistics[2 * dimension:].unbind()
    mmd2 = ((sa.square().sum() - qa) / (n * (n - 1))
            + (sb.square().sum() - qb) / (m * (m - 1))
            - 2 * sa.dot(sb) / (n * m))
    return mmd2, qa / n + qb / m, sa / n - sb / m


def linear_mmd2(a, b):
    """Off-diagonal two-sample MMD estimator, O((N+M)D).

    Unbiased for independent samples from each population. The paper corrupts
    the same clean batch, so its paired cross terms have finite-batch bias;
    we retain the paper-style pairing and document that distinction.
    The estimate may be negative; do not clamp it.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Features must be [N,D] and [M,D] with matching D")
    n, m = len(a), len(b)
    if min(n, m) < 2:
        raise ValueError("Off-diagonal MMD requires at least two samples per set")
    sa, sb = a.sum(dim=0), b.sum(dim=0)
    aa = (sa.square().sum() - a.square().sum()) / (n * (n - 1))
    bb = (sb.square().sum() - b.square().sum()) / (m * (m - 1))
    ab = sa.dot(sb) / (n * m)
    return aa + bb - 2 * ab


def feature_loss(model, clean, noisy, t, lambda_l2=0.1, lambda_gp=1.0,
                 create_graph=True, mmd_scope="local"):
    # All images in each MMD comparison share the same scalar noise level.
    if torch.as_tensor(t).numel() != 1:
        raise ValueError("Use one shared noise level per distribution comparison")
    h_clean, h_noisy = model(clean, t), model(noisy, t)
    if mmd_scope not in ("local", "global"):
        raise ValueError("mmd_scope must be local or global")
    statistics = linear_statistics(h_noisy, h_clean)
    if mmd_scope == "global" and dist.is_initialized() and dist.get_world_size() > 1:
        # Exchange compact sufficient statistics, not images or full Jacobians.
        # Every rank uses one global MMD and global reference feature means.
        statistics = _AllReduceSum.apply(statistics)
    mmd2, l2, mean_difference = discrepancy_from_statistics(statistics, h_clean.shape[1])
    gp = clean.new_zeros(())
    grad_norm = clean.new_zeros(())
    if lambda_gp:
        mixing = torch.rand((len(clean), 1, 1, 1), device=clean.device)
        query = (mixing * noisy + (1 - mixing) * clean).detach().requires_grad_(True)
        witness = (model(query, t) * mean_difference).sum(dim=1)
        gradient = torch.autograd.grad(
            witness.sum(), query, create_graph=create_graph, retain_graph=create_graph
        )[0]
        norms = gradient.flatten(1).norm(dim=1)
        gp = (norms - 1).square().mean()
        grad_norm = norms.mean()
    # Main-text Eq. 11: maximize MMD while minimizing the penalties.
    loss = -mmd2 + lambda_l2 * l2 + lambda_gp * gp
    metrics = {
        "loss": loss.detach(), "mmd2": mmd2.detach(), "l2": l2.detach(),
        "gp": gp.detach(), "witness_grad_norm": grad_norm.detach(),
    }
    return loss, metrics
