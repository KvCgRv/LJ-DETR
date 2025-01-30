import torch  
import torch.nn as nn  
from torch.autograd import Function

class DeformableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1):
        super(DeformableConv2D, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.offset_conv = nn.Conv2d(in_channels, deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()

        self.conv_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

    def forward(self, x):
        offsets = self.offset_conv(x)
        return deform_conv2d(x, offsets, self.conv_weight, self.stride, self.padding, self.dilation, self.deformable_groups)

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, deformable_groups=1):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:])
        ctx.deformable_groups = deformable_groups
        ctx.save_for_backward(input, offset, weight)

        output = deform_conv2d(input, offset, weight, stride, padding, dilation, deformable_groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input, grad_offset, grad_weight = deform_conv2d_backward(input, offset, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.deformable_groups)
        return grad_input, grad_offset, grad_weight, None, None, None, None

deform_conv2d = DeformConvFunction.apply

def deform_conv2d(input, offset, weight, stride=1, padding=0, dilation=1, deformable_groups=1):
    return F.conv2d(input, weight, stride=stride, padding=padding, dilation=dilation)

def deform_conv2d_backward(input, offset, weight, grad_output, stride=1, padding=0, dilation=1, deformable_groups=1):
    grad_input = grad_offset = grad_weight = None
    if ctx.needs_input_grad[0]:
        grad_input = deform_conv2d(input, offset, weight, stride, padding, dilation, deformable_groups)
    if ctx.needs_input_grad[1]:
        grad_offset = torch.zeros_like(offset)
    if ctx.needs_input_grad[2]:
        grad_weight = F.conv2d(input, grad_output, stride=stride, padding=padding, dilation=dilation)
    return grad_input, grad_offset, grad_weight