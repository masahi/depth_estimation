require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

require 'data_make3d'

local height = 480
local width = 352

cutorch.setDevice(2)

model_file = 'model3.lua'
cnn = dofile(model_file)
cnn:cuda()

cudnn.convert(cnn, cudnn)

input = torch.rand(8, 3, height, width):cuda()
--input = torch.rand(8, 3, height, width)

output = cnn:forward(input)
print(output:size())
