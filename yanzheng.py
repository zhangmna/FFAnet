import torch
from torch import nn
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deconv=nn.ConvTranspose2d(3,10,8,2,3)    # 反卷积操作
        #self.pool=nn.MaxPool2d(kernel_size=2)
        #self.add_module()
    def forward(self, x):
        output=self.deconv(x)
        return output
net=Net()
a=torch.rand(10,3,64,64)
a=Variable(a)
out=net(a)
print(out.size())