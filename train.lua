require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data'

local npy4th = require 'npy4th'
cutorch.setDevice(2)

cnn = dofile('model.lua')
cnn:cuda()
criterion = nn.MSECriterion()
criterion.sizeAverage = false
criterion:cuda()

cudnn.convert(cnn, cudnn)

local max_epock = 100

cnn:training()
parameters, gradParameters = cnn:getParameters()


function f(param)
   if param ~= parameters then parameters:copy(x) end
   gradParameters:zero()
   
   local input, gt = load_data()

   input = input:cuda()
   gt = gt:cuda()
   
   local output = cnn:forward(input)

   local loss =  criterion:forward(output, gt)
   local df_do = criterion:backward(output, gt)
   cnn:backward(input, df_do)

   return loss, gradParameters
end

optimState = {
  learningRate = 1e-5,
  weightDecay = 0.0005,
  momentum = 0.9,
  learningRateDecay = 1e-7,
}

function train()
   local n_iter = n_data / batch_size

   loss = 0
   for i = 1, n_iter do
     params, fs = optim.sgd(f, parameters, optimState)
     loss = loss + fs[1]
   end

   return loss / n_iter
   
end

for i = 1, max_epock do
   avg_loss = train()
   print(i, avg_loss)
end

input, gt = load_test_data()

pred = cnn:forward(input:cuda())
npy4th.savenpy('pred.npy', pred)

local checkpoint = {}
model:clearState()
model:float()
cudnn.convert(cnn, nn)
checkpoint.model = cnn
torch.save('model.t7', checkpoint)



