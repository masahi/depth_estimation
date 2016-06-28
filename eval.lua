require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data'

local npy4th = require 'npy4th'

input, gt = load_test_data()
print(input:size())
cutorch.setDevice(2)
cnn = torch.load('model.t7').model
cnn:evaluate()
cnn:cuda()
cudnn.convert(cnn, cudnn)

input = input:cuda()
pred = torch.Tensor(gt:size())
print(pred:size())

for i=1,pred:size()[1] do
   inp = input[i]:reshape(1,3,240, 320)
   --print(i, pred:size())
   out = cnn:forward(inp)
   --print(out:size())
   pred[i] = out[1][1]:float()
end

npy4th.savenpy('pred.npy', pred)

