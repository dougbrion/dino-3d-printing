import torch
import torch.nn.functional as F

class HLoss:
    def __init__(self, temperature_t: float, temperature_s: float):
        self.temperature_t = temperature_t
        self.temperature_s = temperature_s
        
    def __call__(
        self, 
        t: torch.FloatTensor, 
        s: torch.FloatTensor, 
        center: torch.FloatTensor
    ) -> torch.FloatTensor:
        t = F.softmax((t.detach() - center) / self.temperature_t, dim=1)
        log_s = F.log_softmax(s / self.temperature_s, dim=1)

        return -(t * log_s).sum(dim=1).mean()