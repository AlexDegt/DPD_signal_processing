import torch
import torch.nn as nn

class LinearSpline2D(nn.Module):
    """
    Rectangular 2D linear spline basis on [0, 1] x [0, 1].

    Expected input:
        input.shape == [batch, 2, num_points]

    Input is complex-valued, but the basis is built using |input|,
    assuming |input| is normalized to [0, 1].

    Output:
        approx.shape == [batch, 1, num_points]
    """

    def __init__(self, order=4, dtype=torch.complex128, device='cuda:0'):
        super().__init__()

        assert isinstance(order, int) or (isinstance(order, list) and len(order) == 2), \
            "order parameter must be int or list of 2 ints."

        if isinstance(order, int):
            self.order = [order, order]
        else:
            self.order = order

        assert self.order[0] >= 2 and self.order[1] >= 2, \
            "For linear spline basis, each order must be at least 2."

        self.dtype = dtype
        self.device = device
        self.vand = None

        param_num = self.order[0] * self.order[1]
        self.weight = nn.Parameter(
            torch.zeros(param_num, dtype=dtype, device=device),
            requires_grad=True
        )

        self.weight.data = 1.e-2 * (
            torch.rand(param_num, dtype=dtype, device=device)
            + 1j * torch.rand(param_num, dtype=dtype, device=device)
            - 1/2 - 1j/2
        )

    def _build_vandermonde(self, input):
        """
        Build dense basis matrix of shape [batch, num_points, order[0] * order[1]].
        Basis is built from abs(input), assuming abs(input) is in [0, 1].
        """
        x = torch.abs(input) * 1

        batch_size, _, num_points = x.shape
        device = x.device
        real_dtype = x.dtype

        x0 = x[:, 0, :]   # [B, T]
        x1 = x[:, 1, :]   # [B, T]

        n0 = self.order[0]
        n1 = self.order[1]

        # Uniform grid on [0, 1]
        h0 = 1.0 / (n0 - 1)
        h1 = 1.0 / (n1 - 1)

        # Cell coordinates
        u0 = x0 / h0
        u1 = x1 / h1

        # Clamp so x=1 falls into the last cell
        u0 = torch.clamp(u0, 0.0, n0 - 1 - 1e-12)
        u1 = torch.clamp(u1, 0.0, n1 - 1 - 1e-12)

        # Left node indices
        i0 = torch.floor(u0).long()
        i1 = torch.floor(u1).long()

        i0 = torch.clamp(i0, 0, n0 - 2)
        i1 = torch.clamp(i1, 0, n1 - 2)

        # Local coordinates in cell
        d0 = u0 - i0.to(real_dtype)
        d1 = u1 - i1.to(real_dtype)

        # 1D linear basis
        w0_left  = 1.0 - d0
        w0_right = d0
        w1_left  = 1.0 - d1
        w1_right = d1

        # 2D tensor-product basis
        w00 = w0_left  * w1_left
        w10 = w0_right * w1_left
        w01 = w0_left  * w1_right
        w11 = w0_right * w1_right

        # Flattened indices: idx = ix * n1 + iy
        idx00 = i0 * n1 + i1
        idx10 = (i0 + 1) * n1 + i1
        idx01 = i0 * n1 + (i1 + 1)
        idx11 = (i0 + 1) * n1 + (i1 + 1)

        vand = torch.zeros(
            batch_size, num_points, n0 * n1,
            dtype=self.dtype, device=device
        )

        vand.scatter_add_(2, idx00.unsqueeze(-1), w00.to(self.dtype).unsqueeze(-1))
        vand.scatter_add_(2, idx10.unsqueeze(-1), w10.to(self.dtype).unsqueeze(-1))
        vand.scatter_add_(2, idx01.unsqueeze(-1), w01.to(self.dtype).unsqueeze(-1))
        vand.scatter_add_(2, idx11.unsqueeze(-1), w11.to(self.dtype).unsqueeze(-1))

        return vand

    def forward(self, input):
        self.vand = self._build_vandermonde(input)
        approx = (self.vand @ self.weight)[:, None, :]
        return approx

    def get_jacobian(self, input):
        jacobian = self._build_vandermonde(input)[0, ...]
        return jacobian