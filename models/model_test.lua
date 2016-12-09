require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

local height = 224
local width = 288

cutorch.setDevice(1)

model_file = 'model_resnet.lua'

cnn = dofile(model_file)
cnn:float()
params, gradParams = cnn:getParameters()
print(params:size())

cnn:cuda()

cudnn.convert(cnn, cudnn)

input = torch.rand(1, 3, height, width):cuda()

output = cnn:forward(input)
print(output:size())
