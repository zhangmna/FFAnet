import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19
from torch.autograd import Variable
import settings
class SEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        mid = int(input_dim / 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(input_dim, mid), nn.ReLU(inplace=True), nn.Linear(mid, input_dim),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class NoSEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

    def forward(self, x):
        return x


SE = SEBlock if settings.use_se else NoSEBlock

class Batchnormal(nn.Module):
    def __init__(self,channel):
        super(Batchnormal, self).__init__()
        self.bn = nn.BatchNorm2d(channel)
    def forward(self, x):
        bn = self.bn(x)
        return bn
class Non_Batchnormal(nn.Module):
    def __init__(self,channel):
        super(Non_Batchnormal, self).__init__()
    def forward(self, x):
        bn = x
        return bn
BN = Batchnormal if settings.use_bn else Non_Batchnormal
class Tree_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tree_3, self).__init__()
        if settings.dilation is True:
            self.path1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel),nn.LeakyReLU(0.2))
            self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3),BN(out_channel),nn.LeakyReLU(0.2))
            self.path5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5),BN(out_channel), nn.LeakyReLU(0.2))
            self.cat1_3 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),nn.LeakyReLU(0.2))
            self.cat3_5 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),SE(out_channel))
        else:
            self.path1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel), nn.LeakyReLU(0.2))
            self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel), nn.LeakyReLU(0.2))
            self.path5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel), nn.LeakyReLU(0.2))
            self.cat1_3 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), nn.LeakyReLU(0.2))
            self.cat3_5 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), SE(out_channel))

    def forward(self, x):
        path1 = self.path1(x)
        path3 = self.path3(x)
        path5 = self.path5(x)
        cat1_3 = self.cat1_3(torch.cat([path1, path3], dim=1))
        cat3_5 = self.cat3_5(torch.cat([path3, path5], dim=1))
        final = self.cat_final(torch.cat([cat1_3, cat3_5], dim=1))
        return path1,path3,path5,final

class Tree_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tree_2, self).__init__()
        if settings.dilation is True:
            self.path1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel),nn.LeakyReLU(0.2))
            self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3),BN(out_channel),nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),SE(out_channel))
        else:
            self.path1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel), nn.LeakyReLU(0.2))
            self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel), nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), SE(out_channel))

    def forward(self, x):
        path1 = self.path1(x)
        path3 = self.path3(x)
        final = self.cat_final(torch.cat([path1, path3], dim=1))
        return final
class Tree_4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tree_4, self).__init__()
        if settings.dilation is True:
            self.path1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1),BN(out_channel),nn.LeakyReLU(0.2))
            self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3),BN(out_channel),nn.LeakyReLU(0.2))
            self.path5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5),BN(out_channel), nn.LeakyReLU(0.2))
            self.path7 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), BN(out_channel),nn.LeakyReLU(0.2))
            self.cat1_3 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),nn.LeakyReLU(0.2))
            self.cat5_7 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1),SE(out_channel))
        else:
            self.path1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1, 1), BN(out_channel),
                                       nn.LeakyReLU(0.2))
            self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3), BN(out_channel),
                                       nn.LeakyReLU(0.2))
            self.path5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), BN(out_channel),
                                       nn.LeakyReLU(0.2))
            self.path7 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), BN(out_channel),
                                       nn.LeakyReLU(0.2))
            self.cat1_3 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), nn.LeakyReLU(0.2))
            self.cat5_7 = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, 1, 1), SE(out_channel))

    def forward(self, x):
        path1 = self.path1(x)
        path3 = self.path3(x)
        path5 = self.path5(x)
        path7 = self.path7(x)
        cat1_3 = self.cat1_3(torch.cat([path1, path3], dim=1))
        cat5_7 = self.cat3_5(torch.cat([path5, path7], dim=1))
        final = self.cat_final(torch.cat([cat1_3, cat5_7], dim=1))
        return final
class No_tree(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(No_tree, self).__init__()
    def forward(self, x):
        y=x
        return y
if settings.dilation_num==2:
    Tree_use=Tree_2
elif settings.dilation_num==3:
    Tree_use = Tree_3
elif settings.dilation_num==4:
    Tree_use = Tree_4
class Pyramid_2(nn.Module):
    def __init__(self,in_channel,out_channel,dilation):
        super(Pyramid_2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = Tree_use(self.in_channel, self.out_channel)
        #self.conv2 = Tree_use(self.in_channel, self.out_channel)
        #self.conv4 = Tree_use(self.in_channel, self.out_channel)
        #self.conv8 = Tree_use(self.in_channel, self.out_channel)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.pool8 = nn.MaxPool2d(8, 8)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(2*self.out_channel, self.out_channel, 1, 1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        pool1 = self.conv1(self.pool1(x))
        pool2 = self.conv1(self.pool2(x))
        #pool8 = self.conv1(self.pool8(x))
        pool1 = F.upsample_bilinear(pool1, scale_factor=1)
        pool2 = F.upsample_bilinear(pool2, scale_factor=2)
        #pool8 = F.upsample_bilinear(pool8, scale_factor=8)
        out = self.cat_conv(torch.cat([pool1,pool2],dim=1))
        return out
class Pyramid_4(nn.Module):
    def __init__(self,in_channel,out_channel,dilation):
        super(Pyramid_4, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = Tree_use(self.in_channel, self.out_channel)
        #self.conv2 = Tree_use(self.in_channel, self.out_channel)
        #self.conv4 = Tree_use(self.in_channel, self.out_channel)
        #self.conv8 = Tree_use(self.in_channel, self.out_channel)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.pool8 = nn.MaxPool2d(8, 8)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(4*self.out_channel, self.out_channel, 1, 1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        pool1 = self.conv1(self.pool1(x))
        pool2 = self.conv1(self.pool2(x))
        pool4 = self.conv1(self.pool4(x))
        pool8 = self.conv1(self.pool8(x))
        pool1 = F.upsample_bilinear(pool1, scale_factor=1)
        pool2 = F.upsample_bilinear(pool2, scale_factor=2)
        pool4 = F.upsample_bilinear(pool4, scale_factor=4)
        pool8 = F.upsample_bilinear(pool8, scale_factor=8)
        out = self.cat_conv(torch.cat([pool1,pool2,pool4,pool8],dim=1))
        return out
class Pyramid_3(nn.Module):
    def __init__(self,in_channel,out_channel,dilation):
        super(Pyramid_3, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = Tree_use(self.in_channel, self.out_channel)
        #self.conv2 = Tree_use(self.in_channel, self.out_channel)
        #self.conv4 = Tree_use(self.in_channel, self.out_channel)
        #self.conv8 = Tree_use(self.in_channel, self.out_channel)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)
        #self.pool8 = nn.MaxPool2d(8, 8)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(3*self.out_channel, self.out_channel, 1, 1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p4 = self.pool4(x)
        path1, path3, path5, pool1 = self.conv1(p1)
        _,_,_,pool2 = self.conv1(p2)
        _, _, _,pool4 = self.conv1(p4)
        #pool8 = self.conv1(self.pool8(x))
        pool1 = F.upsample_bilinear(pool1, scale_factor=1)
        pool2 = F.upsample_bilinear(pool2, scale_factor=2)
        pool4 = F.upsample_bilinear(pool4, scale_factor=4)
        #pool8 = F.upsample_bilinear(pool8, scale_factor=8)
        out = self.cat_conv(torch.cat([pool1,pool2,pool4],dim=1))

        return out,path1, path3, path5, pool1
if settings.pyramid_num==2:
    My_unit=Pyramid_2
elif settings.pyramid_num==3:
    My_unit = Pyramid_3
elif settings.pyramid_num==4:
    My_unit = Pyramid_4
class My_blocks(nn.Module):
    def __init__(self,in_channel):
        super(My_blocks, self).__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel
        self.num = settings.res_block_num
        self.res = nn.ModuleList()
        self.cat_1 = nn.ModuleList()
        self.cat_2 = nn.ModuleList()
        self.cat_dense=nn.ModuleList()
        for j in range(self.num):
            self.res.append(My_unit(self.in_channel, self.out_channel,2**j))
        if settings.connection_style == 'dense_connection':
            for i in range(self.num+1):
                self.cat_dense.append(nn.Conv2d((i+1)*self.out_channel,self.out_channel,1,1))
        if settings.connection_style == 'multi_short_skip_connection':
            for _ in range(int(self.num/2)-1):
                self.cat_1.append(nn.Conv2d(2*self.in_channel,self.out_channel,1,1))
            for i in range(int(self.num/2)):
                self.cat_2.append(nn.Conv2d((i+2)*self.in_channel,self.out_channel,1,1))
        elif settings.connection_style == 'symmetric_connection':
            for _ in range(int(self.num/2)):
                self.cat_2.append(nn.Conv2d(2*self.in_channel,self.out_channel,1,1))
    def forward(self, x):
        if settings.connection_style == 'dense_connection':
            out=[]
            out.append(x)
            for i in range(self.num):
                if i==0:
                    x,path1, path3, path5, pool1=self.res[i](x)
                else:
                    x, _,_,_ ,_= self.res[i](x)
                out.append(x)
                mid = []
                #print(out[-1].size())
                for j in range(i+2):
                    mid.append(out[j])
                x=self.cat_dense[i+1](torch.cat(mid, dim=1))
            return x,path1, path3, path5, pool1
        if settings.connection_style == 'multi_short_skip_connection':
            out=[]
            out.append(x)
            for i in range(self.num):
                x=self.res[i](x)
                out.append(x)
                if i%2==0 & i>=2:
                    odd=[] #odd：奇数
                    for j in range(i):
                        odd.append(out[2*j+1])
                    x=self.cat_1[int((i-2)/2)](torch.cat(odd,dim=1))
                if i%2==1:
                    even=[]
                    even.append(out[0])
                    even.append(out[2])
                    if i>=3:
                        for s in range(int((i-1)/2)):
                            even.append(out[2*(s+2)])
                    x=self.cat_2[int((i-1)/2)](torch.cat(even,dim=1))
            return x
        elif settings.connection_style == 'symmetric_connection':
            out=[]
            out.append(x)
            for i in range(self.num):
                x=self.res[i](x)
                out.append(x)
                if i >= (int(self.num/2)):
                    x=self.cat_2[int(i-int(self.num/2))](torch.cat([out[-1],out[(-2)*(i-int(self.num/2)+1)-1]],dim=1))
            return x
        elif settings.connection_style == 'no_connection':
            for i in range(self.num):
                x=self.res[i](x)
            return x
class RESCAN(nn.Module):
    def __init__(self):
        super(RESCAN, self).__init__()
        channel_num = settings.feature_map_num
        self.extract = nn.Sequential(
            nn.Conv2d(3,channel_num,3,1,1),
            nn.LeakyReLU(0.2),
            SE(channel_num)
        )
        self.dense = My_blocks(channel_num)
        self.exit = nn.Sequential(
            nn.Conv2d(channel_num,channel_num,3,1,1),
            nn.LeakyReLU(0.2),
            SE(channel_num),
            nn.Conv2d(channel_num, 3, 1, 1)
        )
    def forward(self, x):
        extract=self.extract(x)
        res,path1, path3, path5, pool1 = self.dense(extract)
        final_out = self.exit(res)
        out=[]
        out.append(x-final_out)
        return out,path1, path3, path5, pool1
class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(3,6,8,11), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features

#ts = torch.Tensor(16, 3, 64, 64).cuda()
#vr = Variable(ts)
#net = RESCAN().cuda()
#print(net)
#oups = net(vr)
#for oup in oups:
#    print(oup.size())

