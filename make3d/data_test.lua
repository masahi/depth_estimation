require 'nnx'

matio = require 'matio'

npy4th = require 'npy4th'
depths = npy4th.loadnpy('train_depths.npy')
images = npy4th.loadnpy('train_imgs.npy')

n_data = depths:size()[1]
idx = 1
batch_size = 8

input_width = 345
input_height = 460

output_width = 345
output_height = 460

input_resample = nn.SpatialReSampling{owidth=input_width,oheight=input_height}
output_resample = nn.SpatialReSampling{owidth=output_width,oheight=output_height}

function load_data()

   if idx + batch_size > n_data then
      idx = 1
   end

    _input = images[{{idx, idx+batch_size-1}}]
    _gt = depths[{{idx, idx+batch_size-1}}]:reshape(batch_size, 1, depths:size()[2], depths:size()[3])

    input = torch.Tensor(batch_size, 3, input_height, input_width)
    gt = torch.Tensor(batch_size, 1, output_height, output_width)
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
   test_depths = depths:index(1, test_idx:long())
   test_images = images:index(1, test_idx:long())

  -- test_depths = depths:index(1, train_idx:long())
  -- test_images = images:index(1, train_idx:long())
   
   return input_resample:forward(test_images:double()),
          output_resample:forward(test_depths:double())
end

