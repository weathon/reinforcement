import tqdm
import torch
from torch.distributions.categorical import Categorical
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(39, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 25, kernel_size=3),
        )
        self.options = torch.arange(0.5, 3, 0.1).cuda()
    
    def map(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        sampler = Categorical(logits=tensor) 
        indices = sampler.sample()
        values = self.options[indices]
        values_upsampled = torch.nn.functional.interpolate(
            values.unsqueeze(1), size=(64, 64), mode="bicubic", align_corners=False
        )
        return values_upsampled.squeeze(1), indices
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, maps, step):
        maps = maps.unsqueeze(0).float()
        # print(maps.shape)
        step = torch.tensor(step).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        maps = torch.cat([maps, step.expand(maps.shape[0], 1, maps.shape[2], maps.shape[3])], dim=1)
        return self.conv(maps)
