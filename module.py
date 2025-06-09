import tqdm
import torch
from torch.distributions.categorical import Categorical
class Module(torch.nn.Module):
    def __init__(self, head=8):
        super().__init__()
        self.latent_proj = torch.nn.Linear(1536, 128)
        self.prompt_proj = torch.nn.Linear(1536, 128)
        self.head = head 
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(head, 32, kernel_size=3, padding=1), #why we need padding? alignment?
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 20, kernel_size=3, padding=1),
        )
        self.options = torch.arange(0, 2, 0.1).cuda()
    
    def map(self, tensor, t=5):
        # tensor [BATCH, options, 32, 32]
        tensor = tensor.permute(0, 2, 3, 1)  # [BATCH, 32, 32, options]
        sampler = Categorical(logits=tensor/t) 
        indices = sampler.sample()
        values = self.options[indices]
        values_upsampled = torch.nn.functional.interpolate(
            values.unsqueeze(1), size=(128, 128), mode="bicubic", align_corners=False
        )
        return values_upsampled.squeeze(1), indices
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, latent, prompt):
        # print(prompt.shape, latent.shape) 
        batch_size = latent.shape[0]
        latent = self.latent_proj(latent)
        prompt = self.prompt_proj(prompt)
        latent = latent.view(batch_size, -1, self.head, 128 // self.head)
        prompt = prompt.view(batch_size, -1, self.head, 128 // self.head)
        assert latent.shape[-2:] == prompt.shape[-2:]
        attention = torch.einsum("blhd,bnhd->bhln", latent, prompt).mean(-1)
        attention = attention.view(-1, self.head, 64, 64)
        res = self.conv(attention)
        return res