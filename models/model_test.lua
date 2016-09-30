require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

local height = 224
local width = 288

cutorch.setDevice(1)

model_file = 'model_novgg.lua'
cnn = dofile(model_file)
cnn = cnn:cuda()

cudnn.convert(cnn, cudnn)

input = torch.rand(1, 3, height, width):cuda()

output = cnn:forward(input)
print(output:size())
