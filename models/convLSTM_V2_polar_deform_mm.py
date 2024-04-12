import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision.ops import deform_conv2d
#from .dcn import DeformableConv2d as DeformConv2d
import pdb

class PolarConv(nn.Conv2d):
    def __init__(self, *args, polar_conv_type = None, input_size=None, deform_type="normal", dcnv2=False, dilate_factor=4, **kwargs):
        super(PolarConv, self).__init__(*args, **kwargs)
        self.dcnv2 = dcnv2
        self.polar_conv_type = polar_conv_type
        self.dilate_factor = dilate_factor
        self.deform_type = deform_type
        self.circular_padding = self.padding
        self.padding = 0 
        
        
        if "dilate" in self.polar_conv_type :
            # self.nn_dilate_factor = nn.Parameter(torch.Tensor([self.dilate_factor]))
            if input_size is None:
                self.conv_offset = None
            else:
                self.conv_offset = self.get_offset(input_size)
        
        if "deform" in self.polar_conv_type:
            kh, kw = self.weight.shape[-2:]
            self.offset_conv = nn.Conv2d(self.in_channels, 
                                        2 * kw * kh,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.circular_padding, 
                                        bias=True)
            self.modulator_conv = nn.Conv2d(self.in_channels, 
                                        1 * kw * kh,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.circular_padding, 
                                        bias=True)
            self.tanh = nn.Tanh()

            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            

    def get_offset(self, size):
        h, w = size
        out_h, out_w = h // self.stride[0], w // self.stride[1]
        out_num = out_h * out_w

        kh, kw = self.weight.shape[-2:]
        kernel_num = kh * kw
        grid_x = torch.arange(-((kw - 1) // 2), kw // 2 + 1)
        grid_y = torch.arange(-((kh - 1) // 2), kh // 2 + 1)
        grid_x = grid_x.view(1, kw).repeat(kh, 1)
        grid_y = grid_y.view(kh, 1).repeat(1, kw)
        conv_offset = torch.stack([grid_y, grid_x]).permute(1, 2, 0).contiguous().view(-1)
        conv_offset = conv_offset.view(1, kernel_num, 2).repeat(out_num, 1, 1).type(torch.atan2(torch.tensor(1), torch.tensor(1)).type())
        conv_offset_orig = torch.stack([grid_y, grid_x]).permute(1, 2, 0).contiguous().view(-1)
        conv_offset_orig = conv_offset_orig.view(1, kernel_num, 2).repeat(out_num, 1, 1).type(torch.atan2(torch.tensor(1), torch.tensor(1)).type())
        
        # compute the offset of PolarConv
        h_index = (torch.arange(out_h) + 1) * self.stride[0]
        h_index = h_index - h_index[0]
        w_index = (torch.arange(out_w) + 1) * self.stride[1]
        w_index = w_index - w_index[0] + 1
    
        index_repeated = w_index.unsqueeze(0).repeat(out_h,1).view(-1, 1)
        conv_index = index_repeated + conv_offset[:,:,1]
        conv_index[conv_index<=0] = (w//2) / max(kw//2, 1)
        # offset = w//2 / conv_index # / (max(kw//2, 1))
        offset = (w//2) / (w/2 - (w/2 - conv_index) / self.dilate_factor)
        conv_offset[:,:,0] *= offset
    
        # make out index to inner
        h_index_temp = h_index[:, None, None, None, None].repeat(1, out_w, kh, kw, 2).view(*conv_offset.shape)
        
        conv_offset_temp2 = conv_offset + h_index_temp
        conv_offset -= conv_offset_orig
        conv_offset[conv_offset_temp2 < 0] += h
        conv_offset[conv_offset_temp2 > h-1] -= h
        
        # if stride is bigger than 1, the center of kernel is not on the integer index.s
        shift_h = (h - self.weight.shape[2]) % self.stride[0]
        shift_w = (w - self.weight.shape[3]) % self.stride[1]
        conv_offset[:, :, 0] += shift_w / 2.0
        conv_offset[:, :, 1] += shift_h / 2.0
        
        conv_offset = conv_offset.contiguous().view(1, out_h, out_w, 2 * kernel_num).permute(0, 3, 1, 2)
        
        
        return conv_offset
    
    def circular_padding_func(self, input):
        input = F.pad(input, [0, 0, self.circular_padding[1], self.circular_padding[1]], mode='circular')  
        input = F.pad(input, [self.circular_padding[0], self.circular_padding[0], 0,0], mode='constant', value=0)  
        return input

    def forward(self, input):
        
        if self.polar_conv_type == "normal":
            input = self.circular_padding_func(input)
            out = super().forward(input)
            
        else:
            if "dilate" in self.polar_conv_type :
                if self.conv_offset is None:
                    self.conv_offset = self.get_offset(input.shape[2:])
                    
            if "deform" in self.polar_conv_type:
                self.conv_offset2 = torch.arange(input.shape[3] // self.stride[1]).to(input.device) + 1
                deform_offset = self.offset_conv(input)
                if self.deform_type == "normal":
                    deform_offset = self.tanh(deform_offset)
                elif self.deform_type == "div":
                    deform_offset = self.tanh(deform_offset) /  self.conv_offset2[None, None, None, :]
                    
                if self.dcnv2:
                    mask = self.modulator_conv(input)
                else:
                    mask = None
                    
            if "dilate" in self.polar_conv_type and "deform" in self.polar_conv_type:
                offset = self.conv_offset.to(deform_offset.device) + deform_offset
            elif "deform" in self.polar_conv_type:
                offset = deform_offset
            else:
                offset = self.conv_offset.to(input.device).repeat(len(input), 1, 1, 1)
                mask = None
                
            input = self.circular_padding_func(input)
            out = deform_conv2d(input, offset, self.weight, bias=self.bias, stride=self.stride, mask=mask, padding=self.padding)
                

        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias):
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
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        '''
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        '''

        self.conv = PolarConv(in_channels= self.input_dim + self.hidden_dim, out_channels= 4*hidden_dim, kernel_size= kernel_size, padding=self.padding, polar_conv_type='deform')

        # Motion Gate
        self.input_shape = input_shape
        self.height = 128
        self.width = 128
        self.cnt=0

        kh, kw = 3, 3
        self.offset = torch.rand(4, 2 * kh * kw, self.height, self.width) # [8, 18, 128, 128]
        #self.mask = torch.rand(self.input_shape[0], kh * kw, self.height, self.width).cuda()

        self.deform_conv_gates = PolarConv(in_channels= hidden_dim*2, out_channels= hidden_dim, kernel_size= 3, padding=1, polar_conv_type='deform')
        self.deform_conv_can = PolarConv(in_channels= hidden_dim*3, out_channels= hidden_dim, kernel_size= 3, padding=1, polar_conv_type='deform') # chgd to *3



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
        deform_combined_conv = self.deform_conv_gates(combined, )#offset = self.offset, mask = self.mask) # [8, 320, 128, 128]

        #mcc_i, mcc_f, mcc_o, mcc_g = torch.split(deform_combined_conv, self.hidden_dim, dim=1)
        #mcc_o, mcc_g = torch.split(deform_combined_conv, self.hidden_dim, dim=1)
        #mi = torch.sigmoid(mcc_i) # [8, 80, 128, 128]
        #mf = torch.sigmoid(mcc_f) # [8, 80, 128, 128]

        motion_gate = torch.sigmoid(deform_combined_conv) # [8, 80, 128, 128]
        combined_2 = torch.cat([input_tensor, h_cur, (h_cur-input_tensor)], dim=1) # chgd with motion map

        cc_tnm = self.deform_conv_can(combined_2, )#offset = self.offset, mask = self.mask)
        tnm = torch.tanh(cc_tnm)

        # original
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next) + (motion_gate * tnm)

        #mc_next = mf * c_cur + mi * mg
        #mh_next = mo * torch.tanh(mc_next)

        #c_next += mc_next # chgd
        #h_next += motion_gate * tnm # chgd

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

    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTMV2, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_shape = input_shape
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

            cell_list.append(ConvLSTMCell(input_shape=self.input_shape,
                                        input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

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

