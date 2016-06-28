require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

require 'data'

local height = 240
local width = 320

cutorch.setDevice(1)

model_file = 'model4.lua'
cnn = dofile(model_file)
cnn:cuda()

cudnn.convert(cnn, cudnn)

input = torch.rand(32, 3, height, width):cuda()

output = cnn:forward(input)
print(output:size())
