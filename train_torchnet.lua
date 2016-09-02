local tnt = require 'torchnet'

local model_file = arg[1]
local max_epock = tonumber(arg[2])
local gpu = tonumber(arg[3])
local out_file = arg[4]

local net = dofile(model_file)

require 'MaskedMSECriterion'
local criterion = nn.MaskedMSECriterion()

local engine = tnt.OptimEngine()
local meter  = tnt.AverageValueMeter()

engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   print(state.criterion.output)
end

engine.hooks.onUpdate = function(state)
   state.sample = nil   
   collectgarbage()
   collectgarbage()   
end

engine.hooks.onEndEpoch = function(state)
   mean, std = meter:value()
   print(state.epoch, mean)
   meter:reset()
   
   if state.epoch % 1 == 0 then
     local checkpoint = {}
     state.network:clearState()
     checkpoint.model = state.network
     checkpoint.epoch = state.epoch
     torch.save(out_file, checkpoint)
   end
   
end

require 'cunn'
require 'cutorch'
require 'cudnn'

cutorch.setDevice(gpu)   
net = net:cuda()
criterion = criterion:cuda()
cudnn.convert(net, cudnn)
net:training()

local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
engine.hooks.onSample = function(state)
   igpu:resize(state.sample.input:size() ):copy(state.sample.input)
   tgpu:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = igpu
   state.sample.target = tgpu
end  

require 'data_iterator'
require 'optim'

engine:train{
   network   = net,
   iterator  = getIteratorRaw('train'),
   criterion = criterion,
   maxepoch  = max_epoch,
   optimMethod = optim.adadelta,
   optimState = {
      weightDecay = 0.0005,
      momentum = 0.9,
      learningRateDecay = 1e-7,
   }      
}
