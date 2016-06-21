require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

local width = 224
local height = 224

cutorch.setDevice(2)

cnn = dofile('model.lua')
cnn:cuda()
criterion = nn.MSECriterion()
criterion:cuda()

cudnn.convert(cnn, cudnn)

local max_epock = 200

cnn:training()
parameters, gradParameters = cnn:getParameters()

local function load_data()
   local input = torch.rand(10, 3, height, width):cuda()
   local gt = torch.rand(10, 1, height, width):cuda()

   return input, gt
end

function f(param)
   if param ~= parameters then parameters:copy(x) end

   local input, gt = load_data()

   local output = cnn:forward(input)
   local loss =  criterion:forward(output, gt)
   local df_do = criterion:backward(output, gt)
   cnn:backward(input, df_do)

   return loss, gradParameters
end

optimState = {
  learningRate = 1,
  weightDecay = 0.0005,
  momentum = 0.9,
  learningRateDecay = 1e-7,
}

for i = 1, max_epock do
   print(i)
   optim.sgd(f, parameters, optimState)
end   
