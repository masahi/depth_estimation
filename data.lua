require 'nnx'

local matio = require 'matio'

local npy4th = require 'npy4th'
local depths = npy4th.loadnpy('depths.npy'):transpose(2,3)
local images = npy4th.loadnpy('images.npy'):transpose(3,4)
local splits = matio.load('splits.mat')

local train_idx = splits['trainNdxs']
local test_idx = splits['testNdxs']

local train_idx = train_idx:view(train_idx:nElement())
local test_idx = test_idx:view(test_idx:nElement())

local train_depths = depths:index(1, train_idx:long())
local train_images = images:index(1, train_idx:long())

n_data = train_depths:size()[1]
local idx = 1
batch_size = 8

local input_width = 320
local input_height = 256

local output_width = 320
local output_height = 256

local input_resample = nn.SpatialReSampling{owidth=input_width,oheight=input_height}
local output_resample = nn.SpatialReSampling{owidth=output_width,oheight=output_height}

function load_data()

   if idx + batch_size > n_data then
      idx = 1
   end
   
   local _input = train_images[{{idx, idx+batch_size-1}}]
   local _gt = train_depths[{{idx, idx+batch_size-1}}]:reshape(batch_size, 1, depths:size()[2], depths:size()[3])

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

function load_test_data()
   -- test_depths = depths:index(1, test_idx:long())
   -- test_images = images:index(1, test_idx:long())

  test_depths = depths:index(1, train_idx:long())
  test_images = images:index(1, train_idx:long())
   
   return input_resample:forward(test_images:double()),
          output_resample:forward(test_depths:double())
end

