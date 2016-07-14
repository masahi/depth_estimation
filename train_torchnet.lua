require 'optim'

print('Using torchnet')
local tnt = require 'torchnet'

-- local cmd = torch.CmdLine()
-- cmd:option('-usegpu', true, 'use gpu for training')

-- local config = cmd:parse(arg)
-- print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

-- function that sets of dataset iterator:
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() require 'torchnet' end,
      closure = function()

         local matio = require 'matio'


         local npy4th = require 'npy4th'
         local all_depths = npy4th.loadnpy('depths.npy')
         local all_images = npy4th.loadnpy('images.npy')
         local splits = matio.load('splits.mat')
         
         local train_idx = splits['trainNdxs']
         local test_idx = splits['testNdxs']
         
         local train_idx = train_idx:view(train_idx:nElement())
         local test_idx = test_idx:view(test_idx:nElement())

         local images
         local depths

         if mode == 'train' then
            images = all_images:index(1, train_idx:long())
            depths = all_depths:index(1, train_idx:long())
         else
            images = all_images:index(1, test_idx:long())
            depths = all_depths:index(1, test_idx:long())
         end
         
         n_data = images:size(1)

         print('# of data:', n_data)

         local input_width = 320
         local input_height = 240

         local output_width = 320
         local output_height = 240

         require 'nnx'
         local input_resample = nn.SpatialReSampling{owidth=input_width,oheight=input_height}
         local output_resample = nn.SpatialReSampling{owidth=output_width,oheight=output_height}

         images = input_resample:forward(images:double())
         depths = output_resample:forward(depths:double())

         local n_data = images:size(1)
         
         return tnt.BatchDataset{
            batchsize = 8,
            dataset = tnt.ListDataset{
               list = torch.range(1, n_data):long(),
               load = function(idx)
                  return {
                     input  = images[idx],
                     target = depths[idx]
                  }  
               end,
            }
         }
      end,
   }
end

local model_file = arg[1]
local max_epock = tonumber(arg[2])
local gpu = tonumber(arg[3])
local out_file = arg[4]
local resume = tonumber(arg[5])
local lr = tonumber(arg[6])

-- set up logistic regressor:
local net = dofile(model_file)
local criterion = nn.MSECriterion()

local engine = tnt.OptimEngine()
-- local meter  = tnt.AverageValueMeter()
-- local clerr  = tnt.ClassErrorMeter{topk = {1}}

local loss = 0
local count = 0

engine.hooks.onForwardCriterion = function(state)
   loss = loss + state.criterion.output
   count = count + 1
   -- meter:add(state.criterion.output)
   -- clerr:add(state.network.output, state.sample.target)
   -- if state.training then
   --    print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
   --                        meter:value(), clerr:value{k = 1}))
   -- end
end

engine.hooks.onEndEpoch = function(state)
   avg_loss = loss / count
   print(state.epoch, avg_loss)

   loss = 0
   count = 0
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

engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = criterion,
   maxepoch  = max_epoch,
   optimMethod = optim.adadelta,
   optimState = {
      weightDecay = 0.0005,
      momentum = 0.9,
      learningRateDecay = 1e-7,
   }      
}

-- -- measure test loss and error:
-- meter:reset()
-- clerr:reset()
-- engine:test{
--    network   = net,
--    iterator  = getIterator('test'),
--    criterion = criterion,
-- }
-- print(string.format('test loss: %2.4f; test error: %2.4f',
--                     meter:value(), clerr:value{k = 1}))
