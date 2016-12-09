require 'cutorch'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'optim'

local npy4th = require 'npy4th'
local image = require 'image'
local width = 320
local height = 256
local rgb_mean = 109.31410628

function preprocess(input)
   input = input:float() / 255
   local resnet_mean = {0.485, 0.456, 0.406} 
   local resnet_std = {0.229, 0.224, 0.225}
   for i=1,3 do
      input[1][i]:add(-resnet_mean[i]):div(resnet_std[i])
   end
   return input
end

function eval_test(model_file)

   local input = npy4th.loadnpy('data/nyu/npy/test_images.npy') - rgb_mean
   local gt = npy4th.loadnpy('data/nyu/npy/test_depths.npy')
   
   cutorch.setDevice(1)
   --cnn = torch.load('model3.t7').model
   local cnn = torch.load(model_file).model
   cnn:evaluate()
   cnn:cuda()
   cudnn.convert(cnn, cudnn)
   
   local pred = torch.Tensor(input:size()[1], 480, 640)
   
   for i=1,pred:size()[1] do
      print(i)
      inp = image.scale(input[i], width, height)
      inp = preprocess(inp:reshape(1,3,height,width))
--      inp = inp:reshape(1,3,height,width)      
      --print(i, pred:size())
      out = cnn:forward(inp:cuda())
      --print(out:size())
      pred[i] = image.scale(out[1]:double(), 640, 480)
   end
   
   npy4th.savenpy('pred.npy', pred)
end

function eval_train(model_file)
   local file = io.open("data/nyu/train.txt")
   local file_names = {}
   local count = 1
   for line in file:lines() do
      file_names[count] = 'data/nyu/' .. line
      count = count + 1
   end        
   
   cutorch.setDevice(2)
   
   local cnn = torch.load(model_file).model
   cnn:evaluate()
   cudnn.convert(cnn, cudnn)
   
   n_data = #file_names
   n_eval = 500
   pred = torch.Tensor(n_eval, height, width)
   rgbs = torch.ByteTensor(n_eval, 427, 561, 3)
   depths = torch.Tensor(n_eval, height, width)

   for i=1,n_eval do
      idx = torch.random(1, n_data)
      print(i,file_names[idx] .. '_rgb.npy')      
      local rgb = npy4th.loadnpy(file_names[idx] .. '_rgb.npy')
      local depth = npy4th.loadnpy(file_names[idx] .. '_depth.npy')
      
      rgbs[i] = rgb 
         
      rgb = rgb:permute(3, 1, 2)
      rgb = image.scale(rgb, width, height)
      depth = image.scale(depth, width, height)
      
      depths[i] = depth
   
      out = cnn:forward(preprocess(rgb:reshape(1, 3, height, width)):cuda())
      pred[i] = out[1]:double()
   
   end
   
   npy4th.savenpy('pred_train.npy', pred)
   npy4th.savenpy('rgb_train.npy', rgbs)
   npy4th.savenpy('depth_train.npy', depths)
      
end

eval_train('exp/1021_resnet_droput/445000.t7')
