require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data_make3d'

local npy4th = require 'npy4th'


local model_file = arg[1]
local max_epock = tonumber(arg[2])
local gpu = tonumber(arg[3])
local out_file = arg[4]
local resume = tonumber(arg[5])
local lr = tonumber(arg[6])

cutorch.setDevice(gpu)
print('Learning rate', lr)
if resume == 1 then
   checkpoint = torch.load(model_file)
   cnn = checkpoint.model
   iter_begin = checkpoint.iter + 1
   print('resuming from iter', iter_begin)   
else
   print('training from scratch') 
   cnn = dofile(model_file)
   iter_begin = 1
end  

cnn:cuda()
criterion = nn.MSECriterion()
criterion.sizeAverage = false
criterion:cuda()

cudnn.convert(cnn, cudnn)

cnn:training()
parameters, gradParameters = cnn:getParameters()

print('Number of parameters:', parameters:size()[1])


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
  learningRate = lr,
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

for i = iter_begin, max_epock do
--i = 1
--while true do
   avg_loss = train()
   print(i, avg_loss)

  if i % 50 == 0 then
    local checkpoint = {}
    cnn:clearState()
    checkpoint.model = cnn
    checkpoint.iter = i
    torch.save(out_file, checkpoint)
  end

  i = i + 1
   
end
