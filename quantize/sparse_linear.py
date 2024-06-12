import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer






class SparseLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        sparsity_ratio: float = 0.5,
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.sparsity_ratio = sparsity_ratio
        # de-activate the quantized forward default
        self.sparse = False
        # # initialize quantizer
        # self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        # if not disable_input_quant:
        #     self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        # else:
        #     self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        # initialize quantizer
        self.mask = self.mask_generate(org_module)
    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.sparse:
            # weight = self.weight_quantizer(self.weight)
            # bias = self.bias
            weight = self.weight
            weight.data[self.mask] = 0
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_sparse_state(self, sparse: bool = False):
        self.sparse = sparse

    

    def mask_generate(self, org_module):
        W = org_module.weight.data
        W_metric = torch.abs(W)
        thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*self.sparsity_ratio)].cpu()
        W_mask = (W_metric<=thresh)
        return W_mask




