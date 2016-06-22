require 'nnx'

local npy4th = require 'npy4th'

local depths = npy4th.loadnpy('depths.npy'):transpose(2,3)
local images = npy4th.loadnpy('images.npy'):transpose(3,4)

local n_data = depths:size()[1]
local idx = 1
local batch_size = 10

local input_width = 320
local input_height = 240

local output_width = 40
local output_height = 32

local input_resample = nn.SpatialReSampling{owidth=input_width,oheight=input_height}
local output_resample = nn.SpatialReSampling{owidth=output_width,oheight=output_height}

function load_data()

   if idx + batch_size > n_data then
      idx = 1
   end
   
   local _input = images[{{idx, idx+batch_size-1}}]
   local _gt = depths[{{idx, idx+batch_size-1}}]:reshape(batch_size, 1, depths:size()[2], depths:size()[3])

   local input = torch.Tensor(batch_size, 3, input_height, input_width)
   local gt = torch.Tensor(batch_size, 1, output_height, output_width)
   for i = 1, batch_size do
     input[i] = input_resample:forward(_input[i]:double())
     gt[i] = output_resample:forward(_gt[i]:double())
   end
   
   idx = idx + batch_size
   if idx > n_data then
      idx = 1
   end
   
   return input, gt
end

