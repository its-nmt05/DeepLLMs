import torch


class LinearNoiseScheduler:

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
                            
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        
    def add_noise(self, x, t):
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def sample_timesteps(self, n):
        return torch.randint(1, self.timesteps, (n,))