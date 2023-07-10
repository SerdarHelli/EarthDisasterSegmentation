from transformers import ConvNextV2Config, ConvNextV2Model
import torch
from torch import nn
import torch.nn.functional as tf
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self ,input_dim,out_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class ConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]]=3,
        padding: Union[int, Tuple[int, int], str] = 1,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
        dropout=0.1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU()
        self.dropout=nn.Dropout(dropout)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.activation2(output)
        output=self.dropout(output)

        return output



from transformers import Swinv2Config, Swinv2Model


class SwinChangeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config: Swinv2Config = Swinv2Config.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
        self.swin: Swinv2Model = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window8-256")

        hidden_size=[1024,1024,512,256,128]
        mlps=[]
        for i in range(5):
            mlp = MLP( input_dim=hidden_size[i],out_dim=384)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.diff0 = ConvModule(768,256)
        self.pred0 = ConvModule(256,256)
        self.diff1 = ConvModule(768,256)
        self.pred1 = ConvModule(256,256)
        self.diff2 = ConvModule(768,256)
        self.pred2 = ConvModule(256,256)
        self.diff3 = ConvModule(768,256)
        self.pred3 = ConvModule(256,256)
        self.diff4 = ConvModule(768,256)
        self.pred4 = ConvModule(256,256)

        self.upscale=nn.Sequential(
            ConvModule(256,256,),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ConvModule(256,128,),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        self.linear_fuse = nn.Conv2d(256*5 ,256, kernel_size=1)

        self.classifier = nn.Conv2d(128, 5, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:


        fx1 = self.swin(
            x1,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )
        fx2 = self.swin(
            x2,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )

        features1 = list(fx1.hidden_states)
        features2 = list(fx2.hidden_states)
        batch_size = x1.shape[0]
        features1.reverse()
        features2.reverse()

        for i in range(4):

            features1[i]=features1[i].permute(0, 2, 1)
            features2[i]=features2[i].permute(0, 2, 1)
            height = width = int(math.sqrt(features1[i].shape[-1]))
            features1[i] = self.linear_c[i](
                            features1[i].reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                        )
            
            features1[i] = features1[i].permute(0, 2, 1)
            features1[i] = features1[i].reshape(batch_size, -1, height, width)


            features2[i] = self.linear_c[i](
                  features2[i].reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
              )
            features2[i] = features2[i].permute(0, 2, 1)
            features2[i] = features2[i].reshape(batch_size, -1, height, width)
 



        all_hidden_states=[]
        hidden_states_0=torch.cat([features1[0],features2[0]],axis=1)
        hidden_states_0=self.diff0(hidden_states_0)
        hidden_states_0=self.pred0(hidden_states_0)
        hidden_states_0 = nn.functional.interpolate(hidden_states_0, size=features1[-1].shape[-2:], mode="bilinear", align_corners=False)
        all_hidden_states.append(hidden_states_0)


        hidden_states_1=torch.cat([features1[1],features2[1]],axis=1)
        hid_sub1= nn.functional.interpolate(hidden_states_0, size=features1[1].shape[-2:], mode="bilinear", align_corners=False)
        hidden_states_1=self.diff1(hidden_states_1)+hid_sub1
        hidden_states_1=self.pred1(hidden_states_1)
        hidden_states_1 = nn.functional.interpolate(hidden_states_1, size=features1[-1].shape[-2:], mode="bilinear", align_corners=False)
        all_hidden_states.append(hidden_states_1)

        hidden_states_2=torch.cat([features1[2],features2[2]],axis=1)
        hid_sub2= nn.functional.interpolate(hidden_states_1, size=features1[2].shape[-2:], mode="bilinear", align_corners=False)
        hidden_states_2=self.diff2(hidden_states_2)+hid_sub2
        hidden_states_2=self.pred2(hidden_states_2)
        hidden_states_2 = nn.functional.interpolate(hidden_states_2, size=features1[-1].shape[-2:], mode="bilinear", align_corners=False)
        all_hidden_states.append(hidden_states_2)

        hidden_states_3=torch.cat([features1[3],features2[3]],axis=1)
        hid_sub3= nn.functional.interpolate(hidden_states_2, size=features1[3].shape[-2:], mode="bilinear", align_corners=False)
        hidden_states_3=self.diff3(hidden_states_3)+hid_sub3
        hidden_states_3=self.pred3(hidden_states_3)
        hidden_states_3 = nn.functional.interpolate(hidden_states_3, size=features1[-1].shape[-2:], mode="bilinear", align_corners=False)
        all_hidden_states.append(hidden_states_3)


        hidden_states_4=torch.cat([features1[4],features2[4]],axis=1)
        hid_sub4= nn.functional.interpolate(hidden_states_3, size=features1[4].shape[-2:], mode="bilinear", align_corners=False)
        hidden_states_4=self.diff4(hidden_states_4)+hid_sub4
        hidden_states_4=self.pred4(hidden_states_4)
        hidden_states_4 = nn.functional.interpolate(hidden_states_4, size=features1[-1].shape[-2:], mode="bilinear", align_corners=False)
        all_hidden_states.append(hidden_states_4)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states, axis=1))

        hidden_states=self.upscale(hidden_states)
        logits=self.classifier(hidden_states)

        return logits