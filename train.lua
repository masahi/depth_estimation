require 'cunn'
require 'cutorch'
require 'cudnn'

local tnt = require 'torchnet'

local model_file = arg[1]
local max_epock = tonumber(arg[2])
local gpu = tonumber(arg[3])
local out_dir = arg[4]
local resume = tonumber(arg[5])
local net = dofile(model_file)

require "lfs"
lfs.mkdir(out_dir)
local iter = 1

if resume == 1 then
   local checkpoint = torch.load(arg[6])
   net = checkpoint.model
   iter = tonumber(arg[7])+1
end

require 'MaskedMSECriterion'
local use_log_depth = false
local criterion = nn.MaskedMSECriterion(use_log_depth)
--local criterion = nn.MSECriterion()
local engine = tnt.OptimEngine()
local meter  = tnt.AverageValueMeter()

-- local input_width = 320
-- local input_height = 256
-- local batch_size = 2
local input_width = 288
local input_height = 224
local batch_size = 8
local val_freq = 5000
--local val_freq = 1

local output_width = input_width
local output_height = input_height

local rgb_mean = 109.31410628
local npy4th = require 'npy4th'

local input = npy4th.loadnpy('data/nyu/npy/test_images.npy')
local gt = npy4th.loadnpy('data/nyu/npy/test_depths.npy')
if use_log_depth then
   gt = torch.log(gt)
end   

local image = require('image')

function validate(state)
   state.network:evaluate()

   local n_test = input:size(1)
   local loss = 0

   local timer = torch.Timer()
   for i=1, n_test do
      local rgb = image.scale(input[i]:double(), input_width, input_height)
      local inp = rgb:reshape(1,3,input_height,input_width) - rgb_mean
      local out = state.network:forward(inp:cuda())
      local gt_depth = image.scale(gt[i], input_width, input_height)
      local diff = gt_depth:cuda() - out[1][1]
      loss = loss + diff:pow(2):mean()
   end
--   print('test time: ', timer:time().real)   

   local test_loss = loss / n_test

   local file = io.open("data/nyu/val.txt")

   loss = 0
   local n_val = 0
   timer:reset()
   for line in file:lines() do
     file_name = 'data/nyu/' .. line
     local rgb = npy4th.loadnpy(file_name .. '_rgb.npy')
     local depth = npy4th.loadnpy(file_name .. '_depth.npy')

     if use_log_depth then
        depth = torch.log(depth)
     end

     rgb = rgb:permute(3, 1, 2)     
     rgb = image.scale(rgb:double(), input_width, input_height)
     local inp = rgb:reshape(1,3,input_height,input_width) - rgb_mean
     local out = state.network:forward(inp:cuda())
     local gt_depth = image.scale(depth, input_width, input_height)
     local mask = torch.lt(gt_depth, 0.0001)
     local pred = out[1][1]:double()

     pred[mask] = 0
     gt_depth[mask] = 0
     
     local diff = gt_depth - pred
     loss = loss + diff:pow(2):mean()
     n_val = n_val + 1
   end
--   print('val time: ', timer:time().real)   
   local val_loss = loss / n_val
   
   state.network:training()
   
   return val_loss, test_loss 
end

cutorch.setDevice(gpu)   
net = net:cuda()
criterion = criterion:cuda()
cudnn.convert(net, cudnn)
net:training()

--params, gradParams = net:getParameters()
--print(params:size())

local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
local timer = torch.Timer()
engine.hooks.onSample = function(state)
   igpu:resize(state.sample.input:size() ):copy(state.sample.input - rgb_mean)
   tgpu:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = igpu
   state.sample.target = tgpu
end  

engine.hooks.onForwardCriterion = function(state)
--   print(state.criterion.output)
   meter:add(state.criterion.output)
end

engine.hooks.onBackward = function(state)
end


require 'sys'
engine.hooks.onUpdate = function(state)

   if iter % val_freq == 0 then
--      local timer = torch.Timer()
      val_loss, test_loss = validate(state)
--      print('validation time: ', timer:time().real)
      mean, std = meter:value()
      print(iter, mean, val_loss, test_loss)
      meter:reset()
      
      local checkpoint = {}
      state.network:clearState()
      checkpoint.model = state.network
      checkpoint.iter = state.iter
      checkpoint.val_loss = val_loss

      torch.save(out_dir .. '/' .. iter .. '.t7' , checkpoint)
      
      collectgarbage()
      collectgarbage()
   end
   
   iter = iter + 1
end

require 'data_iterator'
require 'optim'

engine:train{
   network   = net,
   iterator  = get_nyu_full_iterator(input_width, input_height, batch_size),
   criterion = criterion,
   maxepoch  = max_epoch,
   optimMethod = optim.adadelta,
   optimState = {
      weightDecay = 0.0005,
      momentum = 0.9,
      learningRateDecay = 1e-7,
   }      
}
