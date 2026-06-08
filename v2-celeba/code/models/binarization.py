import torch
import torch.nn as nn
import torch.nn.functional as F

class BitConv2d_Std(nn.Conv2d):
    """
    1-bit Weight Convolution with Mean-Centering and scale factors (W1A16 Native).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        w = self.weight
        w_centered = w - w.mean(dim=(1, 2, 3), keepdim=True)
        alpha = w_centered.abs().mean(dim=(1, 2, 3), keepdim=True)
        w_bin = torch.sign(w_centered) * alpha
        w_final = (w_bin - w).detach() + w
        return F.conv2d(x, w_final, self.bias, self.stride, self.padding)

class BinaryActivation_BNN(torch.autograd.Function):
    """
    Custom activation for W1A1: sign function in forward,
    surrogate gradient with straight-through clipping in backward.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Suppress gradients for saturated activations where |x| > 1
        grad_input[input.abs() > 1.0] = 0.0
        return grad_input

class BinaryTanh_BNN(nn.Module):
    def forward(self, x):
        return BinaryActivation_BNN.apply(x)

class BitConv2d_BNN(nn.Conv2d):
    """
    1-bit Weight Convolution without Mean-Centering (W1A1 BNN baseline).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean(dim=(1, 2, 3), keepdim=True)
        w_bin = (w.sign() * alpha - w).detach() + w
        return F.conv2d(x, w_bin, self.bias, self.stride, self.padding)
