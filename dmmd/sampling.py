import torch


@torch.no_grad()
def feature_mean(model, images, t, batch_size=128):
    total = None
    count = 0
    for batch in images.split(batch_size):
        features = model(batch, t)
        summed = features.sum(dim=0)
        total = summed if total is None else total + summed
        count += len(batch)
    if count == 0:
        raise ValueError("Cannot compute feature means from an empty reference set")
    return total / count


def noise_grid(config):
    return torch.linspace(config["t_max"], config["t_min"], config["noise_levels"])


def witness_step(model, particles, t, reference_mean, step_size,
                 noisy_mean=None, pixel_clip=None):
    """Update each query by -eta grad f; do not average f across the batch."""
    with torch.no_grad():
        current_mean = model(particles, t).mean(dim=0) if noisy_mean is None else noisy_mean
        direction = (current_mean - reference_mean).detach()
    with torch.enable_grad():
        query = particles.detach().requires_grad_(True)
        witness = (model(query, t) * direction).sum(dim=1)
        gradient = torch.autograd.grad(witness.sum(), query)[0]
    updated = (query - step_size * gradient).detach()
    if pixel_clip is not None:
        updated.clamp_(-pixel_clip, pixel_clip)
    if not torch.isfinite(updated).all():
        raise FloatingPointError("Non-finite sampler state; inspect features and step_size")
    return updated, gradient.detach()


def sample_particles(model, initial, times, clean_means, steps_per_level,
                     step_size, noisy_means=None, pixel_clip=None, callback=None):
    particles = initial.detach().clone()
    for level, t in enumerate(times):
        t = t.to(particles.device)
        clean_mean = clean_means[level].to(particles.device)
        noisy_mean = None if noisy_means is None else noisy_means[level].to(particles.device)
        for inner in range(steps_per_level):
            particles, gradient = witness_step(
                model, particles, t, clean_mean, step_size, noisy_mean, pixel_clip
            )
            if callback is not None:
                callback(level, inner, t, particles, gradient)
    return particles
