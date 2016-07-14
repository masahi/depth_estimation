require 'cutorch'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data'

local npy4th = require 'npy4th'

input, gt = load_all_data()
print(input:size())
cutorch.setDevice(2)
--cnn = torch.load('model3.t7').model
cnn = torch.load('model_novgg.t7').model
cnn:evaluate()
cnn:cuda()
cudnn.convert(cnn, cudnn)

input = input:cuda()
pred = torch.Tensor(input:size()[1], 480, 640)
print(pred:size())

local output_resample = nn.SpatialReSampling{owidth=640,oheight=480}

for i=1,pred:size()[1] do
   inp = input[i]:reshape(1,3,240, 320)
   --print(i, pred:size())
   out = cnn:forward(inp)
   --print(out:size())
   pred[i] = output_resample:forward(out[1]:double())
end

npy4th.savenpy('pred.npy', pred)

