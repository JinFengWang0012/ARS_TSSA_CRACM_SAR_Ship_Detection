import torch


def rbox_to_gaussian(rboxes: torch.Tensor):
    """Convert rotated boxes (x, y, w, h, theta) to Gaussian parameters."""
    assert rboxes.size(-1) == 5
    xy = rboxes[..., :2]
    wh = rboxes[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    theta = rboxes[..., 4]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rot = torch.stack(
        (cos_theta, -sin_theta, sin_theta, cos_theta), dim=-1).reshape(-1, 2,
                                                                        2)
    half_diag = 0.5 * torch.diag_embed(wh)
    sigma = rot.bmm(half_diag.square()).bmm(rot.transpose(1, 2))
    sigma = sigma.reshape(rboxes.shape[:-1] + (2, 2))
    return xy, sigma


def gaussian_to_cholesky(xy: torch.Tensor, sigma: torch.Tensor):
    """Convert Gaussian mean/covariance to GauCho parameters."""
    chol = torch.linalg.cholesky(sigma)
    alpha = chol[..., 0, 0]
    beta = chol[..., 1, 1]
    gamma = chol[..., 1, 0]
    params = torch.stack((xy[..., 0], xy[..., 1], alpha, beta, gamma), dim=-1)
    return params


def rbox_to_cholesky(rboxes: torch.Tensor):
    """Convert rotated boxes to GauCho parameters (x, y, alpha, beta, gamma)."""
    xy, sigma = rbox_to_gaussian(rboxes)
    return gaussian_to_cholesky(xy, sigma)


def cholesky_to_gaussian(params: torch.Tensor):
    """Convert GauCho parameters (x, y, alpha, beta, gamma) to Gaussian."""
    assert params.size(-1) == 5
    xy = params[..., :2]
    alpha = params[..., 2].clamp_min(1e-7)
    beta = params[..., 3].clamp_min(1e-7)
    gamma = params[..., 4]

    sigma_xx = alpha.square()
    sigma_xy = alpha * gamma
    sigma_yy = beta.square() + gamma.square()
    sigma = torch.stack(
        (sigma_xx, sigma_xy, sigma_xy, sigma_yy), dim=-1).reshape(
            params.shape[:-1] + (2, 2))
    return xy, sigma


def gaussian_to_rbox(xy: torch.Tensor,
                     sigma: torch.Tensor,
                     eps: float = 1e-7) -> torch.Tensor:
    """Convert Gaussian mean/covariance back to rotated boxes."""
    a = sigma[..., 0, 0]
    b = sigma[..., 1, 1]
    c = sigma[..., 0, 1]

    trace = a + b
    diff = torch.sqrt(((a - b).square() + 4 * c.square()).clamp_min(eps))
    lambda_major = ((trace + diff) * 0.5).clamp_min(eps)
    lambda_minor = ((trace - diff) * 0.5).clamp_min(eps)

    width = 2.0 * torch.sqrt(lambda_major)
    height = 2.0 * torch.sqrt(lambda_minor)
    theta = 0.5 * torch.atan2(2 * c, a - b + eps)

    return torch.stack((xy[..., 0], xy[..., 1], width, height, theta), dim=-1)
