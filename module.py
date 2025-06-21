import tqdm
import torch
from torch.distributions.categorical import Categorical
class Module(torch.nn.Module):
    def __init__(self, head=8):
        super().__init__()
        self.latent_proj = torch.nn.Linear(2432, 512)
        self.prompt_proj = torch.nn.Linear(2432, 512) 
        self.head = head
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(head * 2, 128, kernel_size=3), #why we need padding? alignment?
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 25, kernel_size=3), #why this dim will effect speed a lot
        )
        self.time_emb = torch.nn.Embedding(50, head)
        self.options = torch.arange(0.5, 3, 0.1).cuda()
    
    def map(self, tensor):
        # tensor [BATCH, options, 32, 32]
        tensor = tensor.permute(0, 2, 3, 1)  # [BATCH, 32, 32, options]
        sampler = Categorical(logits=tensor) 
        indices = sampler.sample()
        values = self.options[indices]
        values_upsampled = torch.nn.functional.interpolate(
            values.unsqueeze(1), size=(64, 64), mode="bicubic", align_corners=False
        )
        return values_upsampled.squeeze(1), indices
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, latent, prompt, step):
        # print(prompt.shape, latent.shape) 
        batch_size = latent.shape[0]
        latent = self.latent_proj(latent)
        prompt = self.prompt_proj(prompt)
        latent = latent.view(batch_size, -1, self.head, 512 // self.head)
        prompt = prompt.view(batch_size, -1, self.head, 512 // self.head)
        assert latent.shape[-2:] == prompt.shape[-2:]
        attention = torch.einsum("blhd,bnhd->bhln", latent, prompt).mean(-1) 
        time_embed = self.time_emb(torch.tensor(step).cuda()).view(batch_size, self.head, 1, 1).repeat(1, 1, 32, 32)
        attention = attention.view(-1, self.head, 32, 32)
        attention = torch.cat([attention, time_embed], dim=1)  # [BATCH, head * 2, 32, 32]
        res = self.conv(attention)
        # global avg pooling
        # res = res.mean(dim=[2, 3], keepdim=True)
        return res