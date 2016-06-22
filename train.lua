require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

require 'data'

cutorch.setDevice(2)

cnn = dofile('model.lua')
cnn:cuda()
criterion = nn.MSECriterion()
criterion:cuda()

cudnn.convert(cnn, cudnn)

local max_epock = 50

cnn:training()
parameters, gradParameters = cnn:getParameters()


function f(param)
   if param ~= parameters then parameters:copy(x) end

   local input, gt = load_data()

   input = input:cuda()
   gt = gt:cuda()
   
   local output = cnn:forward(input)

   local loss =  criterion:forward(output, gt)
   local df_do = criterion:backward(output, gt)
   cnn:backward(input, df_do)

   print(loss)
   return loss, gradParameters
end

optimState = {
  learningRate = 1e-10,
  weightDecay = 0.0005,
  momentum = 0.9,
  learningRateDecay = 1e-7,
}

function train()
   for i = 1, 100 do
     optim.sgd(f, parameters, optimState)
   end  
end

for i = 1, max_epock do
   print(i)
   train()
end

torch.save('model.t7', cnn:clearState())
