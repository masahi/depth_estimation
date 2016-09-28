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

function get_nyu_full_iterator()
   return tnt.ParallelDatasetIterator{
      nthread = 4,
      init    = function() require 'torchnet' end,
      closure = function()
         
         local file = io.open("data/nyu/train.txt")
         local file_names = {}
         local count = 1
         for line in file:lines() do
            file_names[count] = 'data/nyu/' .. line
            count = count + 1
         end        
         
         local n_data = #file_names
         
         local input_width = 288
         local input_height = 224
      
         local output_width = input_width
         local output_height = input_height
      
         require 'nnx'
         local npy4th = require 'npy4th'
         local image = require 'image'

         local randomkit = require 'randomkit'
         
         return tnt.BatchDataset{
            batchsize = 8,

            dataset = tnt.ListDataset{
               list = torch.range(1, n_data):long(),
               load = function(idx)
                  local rgb = npy4th.loadnpy(file_names[idx] .. '_rgb.npy')
                  local depth = npy4th.loadnpy(file_names[idx] .. '_depth.npy')

                  rgb = rgb:permute(3, 1, 2):double()

                  if randomkit.randint(1,2) == 1 then                  
                    rgb = image.hflip(rgb)
                    depth = image.hflip(depth)
                  end
                  
                  rgb = image.scale(rgb, input_width, input_height)
                  depth = image.scale(depth, output_width, output_height)
                  
                  return {
                      input=rgb,
                      target=depth
                  }                    
               end,
            }
         }
      end,
   }
end
