from typing import Union, List

import tree
import torch
from torch import nn
import numpy as np

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        layers: List = [256, 256],
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    in_size = input_size
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    mlp_layers = []
    for size in layers:
        mlp_layers.append(nn.Linear(in_size, size))
        mlp_layers.append(activation)
        in_size = size
    mlp_layers.append(nn.Linear(in_size, output_size))
    mlp_layers.append(output_activation)
    return nn.Sequential(*mlp_layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

def init_weights(m: nn.Module, gain: float = 1):
    """
        Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(m.weight, gain=gain)
        # torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            m.bias.data.fill_(0.00)

def scale_last_layer(net):
    last_layer = [m for m in net.children()][-2]
    last_layer.weight.data.copy_(0.01 * last_layer.weight.data)

def convert_to_numpy(x, reduce_type=True):
    """Converts values in `stats` to non-Tensor numpy or python types.
    Args:
        x: Any (possibly nested) struct, the values in which will be
            converted and returned as a new struct with all torch/tf tensors
            being converted to numpy types.
        reduce_type: Whether to automatically reduce all float64 and int64 data
            into float32 and int32 data, respectively.
    Returns:
        A new struct with the same structure as `x`, but with all
        values converted to numpy arrays (on CPU).
    """

    # The mapping function used to numpyize torch/tf Tensors (and move them
    # to the CPU beforehand).
    def mapping(item):
        if torch and isinstance(item, torch.Tensor):
            ret = (
                item.cpu().item()
                if len(item.size()) == 0
                else item.detach().cpu().numpy()
            )
        else:
            ret = item
        if reduce_type and isinstance(ret, np.ndarray):
            if np.issubdtype(ret.dtype, np.floating):
                ret = ret.astype(np.float32)
            elif np.issubdtype(ret.dtype, int):
                ret = ret.astype(np.int32)
            return ret
        return ret

    return tree.map_structure(mapping, x)

def convert_to_torch_tensor(x, device):
    """  Copied from ray/rllib
    Converts any struct to torch.Tensors.
    x (any): Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.
    Returns:
        Any: A new struct with the same structure as `stats`, but with all
            values converted to torch Tensor types.
    """

    def mapping(item):
        # Already torch tensor -> make sure it's on right device.
        if torch.is_tensor(item):
            return item if device is None else item.to(device)
        # Numpy arrays.
        if isinstance(item, np.ndarray):
            # Object type (e.g. info dicts in train batch): leave as-is.
            if item.dtype == object:
                return item
            # Non-writable numpy-arrays will cause PyTorch warning.
            elif item.flags.writeable is False:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tensor = torch.from_numpy(item)
            # Already numpy: Wrap as torch tensor.
            else:
                tensor = torch.from_numpy(item)
        # Everything else: Convert to numpy, then wrap as torch tensor.
        else:
            tensor = torch.from_numpy(np.asarray(item))
        # Floatify all float64 tensors.
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor if device is None else tensor.to(device)

    return tree.map_structure(mapping, x)
