import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn.functional as F
import torch.nn as nn

@CONV_LAYERS.register_module()
class AeConv(nn.Conv2d):
    def __init__(self, *args, input_size=None, **kwargs):
        super(AeConv, self).__init__(*args, **kwargs)
        if input_size is None:
            self.ae_offset = None
        else:
            self.ae_offset = self.get_offset(input_size)

    def get_offset(self, size):
        # compute the rotation matrix of AeConv
        h, w = size
        out_h, out_w = h // self.stride[0], w // self.stride[1]
        cart_x = torch.arange(out_w) - out_w / 2.0 + 0.5
        cart_y = -(torch.arange(out_h) - out_w / 2.0 + 0.5)
        cart_x = cart_x.view(1, len(cart_x)).repeat(len(cart_y), 1)
        cart_y = cart_y.view(len(cart_y), 1).repeat(1, len(cart_x))
        azimuth = torch.atan2(cart_x, cart_y)
        rot_matrix = torch.stack([torch.cos(azimuth), torch.sin(azimuth), -torch.sin(azimuth), torch.cos(azimuth)], -1)
        rot_matrix = rot_matrix.view(-1, 2, 2)

        # sampling grid of type convolution
        kh, kw = self.weight.shape[-2:]
        kernel_num = kh * kw
        grid_x = torch.arange(-((kw - 1) // 2), kw // 2 + 1)
        grid_y = torch.arange(-((kh - 1) // 2), kh // 2 + 1)
        grid_x = grid_x.view(1, kw).repeat(kh, 1)
        grid_y = grid_y.view(kh, 1).repeat(1, kw)
        conv_offset = torch.stack([grid_y, grid_x]).permute(1, 2, 0).contiguous().view(-1)

        # compute the offset of AeConv
        conv_offset = conv_offset.view(1, kernel_num, 2).repeat(len(rot_matrix), 1, 1).type(rot_matrix.type())
        ae_offset = torch.bmm(rot_matrix, conv_offset.transpose(1, 2)).transpose(1, 2) - conv_offset

        # align the sampled grid with the feature
        shift_h = (h - self.weight.shape[2]) % self.stride[0]
        shift_w = (w - self.weight.shape[3]) % self.stride[1]
        ae_offset[:, :, 0] += shift_w / 2.0
        ae_offset[:, :, 1] += shift_h / 2.0

        # reshape the offset of AeConv
        ae_offset = ae_offset.contiguous().view(1, azimuth.shape[0], azimuth.shape[1], 2 * kernel_num).permute(0, 3, 1, 2)
        return ae_offset

    def forward(self, input):
        if self.ae_offset is None:
            self.ae_offset = self.get_offset(input.shape[2:])
        ae_offset = self.ae_offset.to(input.device).repeat(len(input), 1, 1, 1)
        
        out = deform_conv2d(input, ae_offset, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return out


@CONV_LAYERS.register_module()
class AeConv_DCN(nn.Conv2d):
    def __init__(self, *args, input_size=None, **kwargs):
        super(AeConv_DCN, self).__init__(*args, **kwargs)
        if input_size is None:
            self.ae_offset = None
        else:
            self.ae_offset = self.get_offset(input_size)
            
        kh, kw = self.weight.shape[-2:]
        self.offset_conv = AeConv(self.in_channels, 
                                    2 * kw * kh,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding, 
                                    bias=True)

    def get_offset(self, size):
        # compute the rotation matrix of AeConv
        h, w = size
        out_h, out_w = h // self.stride[0], w // self.stride[1]
        cart_x = torch.arange(out_w) - out_w / 2.0 + 0.5
        cart_y = -(torch.arange(out_h) - out_w / 2.0 + 0.5)
        cart_x = cart_x.view(1, len(cart_x)).repeat(len(cart_y), 1)
        cart_y = cart_y.view(len(cart_y), 1).repeat(1, len(cart_x))
        azimuth = torch.atan2(cart_x, cart_y)
        rot_matrix = torch.stack([torch.cos(azimuth), torch.sin(azimuth), -torch.sin(azimuth), torch.cos(azimuth)], -1)
        rot_matrix = rot_matrix.view(-1, 2, 2)

        # sampling grid of type convolution
        kh, kw = self.weight.shape[-2:]
        kernel_num = kh * kw
        grid_x = torch.arange(-((kw - 1) // 2), kw // 2 + 1)
        grid_y = torch.arange(-((kh - 1) // 2), kh // 2 + 1)
        grid_x = grid_x.view(1, kw).repeat(kh, 1)
        grid_y = grid_y.view(kh, 1).repeat(1, kw)
        conv_offset = torch.stack([grid_y, grid_x]).permute(1, 2, 0).contiguous().view(-1)

        # compute the offset of AeConv
        conv_offset = conv_offset.view(1, kernel_num, 2).repeat(len(rot_matrix), 1, 1).type(rot_matrix.type())
        ae_offset = torch.bmm(rot_matrix, conv_offset.transpose(1, 2)).transpose(1, 2) - conv_offset

        # align the sampled grid with the feature
        shift_h = (h - self.weight.shape[2]) % self.stride[0]
        shift_w = (w - self.weight.shape[3]) % self.stride[1]
        ae_offset[:, :, 0] += shift_w / 2.0
        ae_offset[:, :, 1] += shift_h / 2.0

        # reshape the offset of AeConv
        ae_offset = ae_offset.contiguous().view(1, azimuth.shape[0], azimuth.shape[1], 2 * kernel_num).permute(0, 3, 1, 2)
        return ae_offset

    def forward(self, input):
        if self.ae_offset is None:
            self.ae_offset = self.get_offset(input.shape[2:])
        ae_offset = self.ae_offset.to(input.device).repeat(len(input), 1, 1, 1)
        
        deform_offset = self.offset_conv(input)
        ae_offset += deform_offset
        
        out = deform_conv2d(input, ae_offset, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return out




 
@CONV_LAYERS.register_module()
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
