require 'cutorch'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data'

local npy4th = require 'npy4th'

local file = io.open("nyu/file_names_shuffled.txt")
local file_names = {}
local count = 1
for line in file:lines() do
   file_names[count] = 'nyu/' .. line
   count = count + 1
end        

cutorch.setDevice(1)

local cnn = torch.load('model_big_par.t7').model
cnn:evaluate()
cudnn.convert(cnn, cudnn)

n_data = #file_names
n_eval = 100
pred = torch.Tensor(n_eval, 200, 280)
rgbs = torch.ByteTensor(n_eval, 427, 561, 3)
depths = torch.Tensor(n_eval, 200, 280)

local input_resample = nn.SpatialReSampling{owidth=280,oheight=200}
local output_resample = nn.SpatialReSampling{owidth=280,oheight=200}

for i=1,n_eval do
   idx = torch.random(1, n_data)
   local rgb = npy4th.loadnpy(file_names[idx] .. '\n_rgb.npy')
   local depth = npy4th.loadnpy(file_names[idx] .. '\n_depth.npy')
   
   rgbs[i] = rgb
      
   rgb = rgb:permute(3, 1, 2)
   rgb = input_resample:forward(rgb:reshape(1, 3, rgb:size(2), rgb:size(3)):double())
   depth = output_resample:forward(depth:reshape(1, depth:size(1), depth:size(2)))
   
   depths[i] = depth[1]

   out = cnn:forward(rgb:cuda())
   pred[i] = out[1]:double()

end

npy4th.savenpy('pred_train.npy', pred)
npy4th.savenpy('rgb_train.npy', rgbs)
npy4th.savenpy('depth_train.npy', depths)

