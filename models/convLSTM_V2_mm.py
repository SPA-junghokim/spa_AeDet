import torch.nn as nn
import torch
import torchvision
from .dcn import DeformableConv2d as DeformConv2d
from mmcv.cnn import build_conv_layer
# from random import randrange

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, 
                    deform_conv_lstm, motion_gate, bevdepth):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        kh, kw = self.kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        if deform_conv_lstm:
            if bevdepth == False:
                self.conv = build_conv_layer(dict(type='AeConv_DCN'), in_channels= self.input_dim + self.hidden_dim, out_channels= 4*hidden_dim, kernel_size= kernel_size, padding=self.padding)
                self.deform_conv_gates = build_conv_layer(dict(type='AeConv_DCN'), in_channels= hidden_dim*2, out_channels= hidden_dim, kernel_size= 3, padding=1)
                self.deform_conv_can = build_conv_layer(dict(type='AeConv_DCN'), in_channels= hidden_dim*3, out_channels= hidden_dim, kernel_size= 3, padding=1)
            else:
                self.conv = DeformConv2d(in_channels= self.input_dim + self.hidden_dim, out_channels= 4*hidden_dim, kernel_size= kernel_size, padding=self.padding)
                self.deform_conv_gates = DeformConv2d(in_channels= hidden_dim*2, out_channels= hidden_dim, kernel_size= 3, padding=1)
                self.deform_conv_can = DeformConv2d(in_channels= hidden_dim*3, out_channels= hidden_dim, kernel_size= 3, padding=1) # chgd to *3
        else:
            if bevdepth == False:
                self.conv = build_conv_layer(dict(type='AeConv'), in_channels= self.input_dim + self.hidden_dim, out_channels= 4*hidden_dim, kernel_size= kernel_size, padding=self.padding)
                self.deform_conv_gates = build_conv_layer(dict(type='AeConv'), in_channels= hidden_dim*2, out_channels= hidden_dim, kernel_size= 3, padding=1)
                self.deform_conv_can = build_conv_layer(dict(type='AeConv'), in_channels= hidden_dim*3, out_channels= hidden_dim, kernel_size= 3, padding=1)
            else:
                self.conv = build_conv_layer(dict(type='Conv2d'), in_channels= self.input_dim + self.hidden_dim, out_channels= 4*hidden_dim, kernel_size= kernel_size, padding=self.padding)
                self.deform_conv_gates = build_conv_layer(dict(type='Conv2d'), in_channels= hidden_dim*2, out_channels= hidden_dim, kernel_size= 3, padding=1)
                self.deform_conv_can = build_conv_layer(dict(type='Conv2d'), in_channels= hidden_dim*3, out_channels= hidden_dim, kernel_size= 3, padding=1)
        self.motion_gate = motion_gate


    def forward(self, input_tensor, cur_state): # input_tensor: [8, 80, 128, 128]
        h_cur, c_cur = cur_state # each [8, 80, 128, 128]

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)  # [8, 160, 128, 128]

        combined_conv = self.conv(combined,) # [8, 320, 128, 128]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i) # [8, 80, 128, 128]
        f = torch.sigmoid(cc_f) # [8, 80, 128, 128]
        o = torch.sigmoid(cc_o) # [8, 80, 128, 128]
        g = torch.tanh(cc_g) # [8, 80, 128, 128]

        # Motion Gate
        if self.motion_gate:
            deform_combined_conv = self.deform_conv_gates(combined, )#offset = self.offset, mask = self.mask) # [8, 320, 128, 128]
            motion_gate = torch.sigmoid(deform_combined_conv) # [8, 80, 128, 128]
            combined_2 = torch.cat([input_tensor, h_cur, (h_cur-input_tensor)], dim=1) # chgd with motion map
            cc_tnm = self.deform_conv_can(combined_2)
            tnm = torch.tanh(cc_tnm)
            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next) + (motion_gate * tnm)
        else:
            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next) 

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device='cuda'),
                torch.zeros(batch_size, self.hidden_dim, height, width, device='cuda'))


class ConvLSTMV2(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, deform_conv_lstm=False, motion_gate=False,
                 bevdepth=False):
        super(ConvLSTMV2, self).__init__()
        # torch.cuda.set_device(randrange(4))
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(
                                        input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias, 
                                          deform_conv_lstm=deform_conv_lstm,
                                          motion_gate=motion_gate,
                                          bevdepth=bevdepth))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers): # 3

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): # 3
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                             cur_state=[h, c])
                

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

