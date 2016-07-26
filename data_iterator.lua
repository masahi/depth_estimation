local tnt = require 'torchnet'

function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() require 'torchnet' end,
      closure = function()

         local matio = require 'matio'
         local npy4th = require 'npy4th'
         local all_depths = npy4th.loadnpy('depths.npy')
         local all_images = npy4th.loadnpy('images.npy')
         local splits = matio.load('splits2.mat')
         
         local images
         local depths

         if mode == 'train' then
            local train_idx = splits['trainNdxs']
            local train_idx = train_idx:view(train_idx:nElement())
            images = all_images:index(1, train_idx:long())
            depths = all_depths:index(1, train_idx:long())
         else
            local test_idx = splits['testNdxs']
            local test_idx = test_idx:view(test_idx:nElement())
            images = all_images:index(1, test_idx:long())
            depths = all_depths:index(1, test_idx:long())
         end

         local n_data = images:size(1)
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

function getIteratorRaw(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() require 'torchnet' end,
      closure = function()
         
         local file = io.open("nyu/file_names_shuffled.txt")
         local file_names = {}
         local count = 1
         for line in file:lines() do
            file_names[count] = 'nyu/' .. line
            count = count + 1
         end        
         
         local n_data = #file_names
         print('# of data:', n_data)

         local input_width = 280
         local input_height = 200

         local output_width = 280
         local output_height = 200

         -- local input_width = 144
         -- local input_height = 104

         -- local output_width = 144
         -- local output_height = 104
         
         require 'nnx'
         local input_resample = nn.SpatialReSampling{owidth=input_width,oheight=input_height}
         local output_resample = nn.SpatialReSampling{owidth=output_width,oheight=output_height}
         local matio = require 'matio'
         
         return tnt.BatchDataset{
            batchsize = 8,
--            batchsize = 32,
            dataset = tnt.ListDataset{
               list = torch.range(1, n_data):long(),
               load = function(idx)
                  local data = matio.load(file_names[idx])
                  local rgb = data['rgb']
                  local depth = data['depth_filled']
                  rgb = rgb:transpose(3, 1, 2)
                  rgb = input_resample:forward(rgb:reshape(1, 3, rgb:size(2), rgb:size(3)):double())
                  depth = output_resample:forward(depth:reshape(1, depth:size(1), depth:size(2)))
                  return {
                     input  = rgb[1],
                     target = depth[1]
                  }  
               end,
            }
         }
      end,
   }
end

local file = io.open("nyu/file_names_shuffled.txt")
local file_names = {}
local count = 1
for line in file:lines() do
   file_names[count] = 'nyu/' .. line
   count = count + 1
end        

local n_data = #file_names
print('# of data:', n_data)

local input_width = 280
local input_height = 200

local output_width = 280
local output_height = 200

require 'nnx'
local input_resample = nn.SpatialReSampling{owidth=input_width,oheight=input_height}
local output_resample = nn.SpatialReSampling{owidth=output_width,oheight=output_height}
local matio = require 'matio'

function getIteratorRaw2(mode)
   return tnt.DatasetIterator{
         dataset =  tnt.BatchDataset{
            batchsize = 8,
            dataset = tnt.ListDataset{
               list = torch.range(1, n_data):long(),
               load = function(idx)
                  local data = matio.load(file_names[idx])
                  local rgb = data['rgb']
                  local depth = data['depth_filled']
                  rgb = rgb:transpose(3, 1, 2)
                  rgb = input_resample:forward(rgb:reshape(1, 3, rgb:size(2), rgb:size(3)):double())
                  depth = output_resample:forward(depth:reshape(1, depth:size(1), depth:size(2)))
                  return {
                     input  = rgb[1],
                     target = depth[1]
                  }  
               end,
            }
         }
   }
end