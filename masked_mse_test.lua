require 'torch'
require 'nn'
require 'MaskedMSECriterion'
require 'cutorch'

masked_mse = nn.MaskedMSECriterion()
mse = nn.MSECriterion()

f_name = 'nyu/training/basement_0001a/15.mat'

npy4th = require 'npy4th'

depth = npy4th.loadnpy(f_name .. '_depth.npy')   

mask = torch.lt(depth, 0.001)

pred = torch.rand(depth:size())

require 'cunn'
masked_mse = masked_mse:cuda()
mse = mse:cuda()
output = mse:forward(pred:cuda(), depth:cuda())
output2 = masked_mse:forward(pred:cuda(), depth:cuda())

output = mse:backward(pred:cuda(), depth:cuda())
output2 = masked_mse:backward(pred:cuda(), depth:cuda())


