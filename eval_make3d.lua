require 'cutorch'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data_make3d'

local npy4th = require 'npy4th'

input, gt = load_test_data()
print(input:size())
cutorch.setDevice(1)
cnn = torch.load('model3_m3d.t7').model
cnn:evaluate()
cnn:cuda()
cudnn.convert(cnn, cudnn)

input = input:cuda()
pred = torch.Tensor(input:size()[1], 460, 345)
print(pred:size())

local output_resample = nn.SpatialReSampling{owidth=460,oheight=345}

for i=1,pred:size()[1] do
   inp = input[i]:reshape(1,3, 480, 352)
   --print(i, pred:size())
   out = cnn:forward(inp)
   --print(out:size())
   pred[i] = output_resample:forward(out[1]:double())
end

npy4th.savenpy('pred_m3d.npy', pred)

