require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

require 'data'

local height = 256
local width = 320

--cutorch.setDevice(2)

model_file = 'model3.lua'
cnn = dofile(model_file)
--cnn:cuda()

--cudnn.convert(cnn, cudnn)

--input = torch.rand(8, 3, height, width):cuda()
input = torch.rand(8, 3, height, width)

output = cnn:forward(input)
print(output:size())
