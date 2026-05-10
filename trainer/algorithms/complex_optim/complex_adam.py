import torch

class ComplexAdam(torch.optim.Optimizer):
    """
        Class implements correct Adam optimizer for complex parameters
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, mode="full"):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        # Parameter which shows whether each parameter has its own adaptive step
        # or only one for every parameter
        assert mode == "simple" or mode == "full", \
            f"Adam \'mode\ must be either \'simple\' or \'full\', but \'{mode}\' is given."
        self.__mode = mode
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['exp_avg'] = torch.zeros_like(p)
                self.state[p]['step'] = 0
                if self.__mode == "full":
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p, dtype=p.real.dtype)
                elif self.__mode == "simple":
                    self.state[p]['exp_avg_sq'] = torch.tensor([0], dtype=p.real.dtype, device=p.device)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, betas, eps = group['lr'], group['betas'], group['eps']
            beta1, beta2 = betas
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad  # Complex gradient
                state = self.state[p]
                
                # Update step
                state['step'] += 1
                t = state['step']

                # Extract 1-st and 2-nd moments
                m, v = state['exp_avg'], state['exp_avg_sq']

                # Update m (1-st moment estimation)
                m[:] = beta1 * m + (1 - beta1) * g

                # Update v (2-nd moment estimation)
                if self.__mode == "full":
                    v[:] = beta2 * v + (1 - beta2) * g.abs().square()
                elif self.__mode == "simple": 
                    v[:] = beta2 * v + (1 - beta2) * g.norm().square()

                # Bias-corrected moments
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Full complex estimation
                p[:] = p - lr * m_hat / (torch.sqrt(v_hat) + eps)