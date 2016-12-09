require 'nn'
require 'cutorch'
require 'cudnn'

local resnet = torch.load('/home/masa/projects/torch/dev/data/resnet-50.t7')

local cnn = nn.Sequential()
for i = 1, 8 do
  local layer = resnet:get(i)
  cnn:add(layer)
end

local deconv_input = 2048
for i =1,5 do
  cnn:add(nn.SpatialFullConvolution(deconv_input, deconv_input/2, 4, 4, 2, 2, 1, 1))
  cnn:add(nn.SpatialBatchNormalization(deconv_input/2)):add(nn.ReLU(true))
   
  cnn:add(nn.SpatialConvolution(deconv_input/2, deconv_input/2, 3, 3, 1,1,1,1))
  cnn:add(nn.SpatialBatchNormalization(deconv_input/2)):add(nn.ReLU(true))
   
  if i < 5 then
    cnn:add(nn.SpatialConvolution(deconv_input/2, deconv_input/2, 3, 3, 1,1,1,1))     
    cnn:add(nn.SpatialBatchNormalization(deconv_input/2)):add(nn.ReLU(true))
  else
    cnn:add(nn.Dropout(0.2))
    cnn:add(nn.SpatialConvolution(deconv_input/2, 1, 3, 3, 1,1,1,1))          
  end
  
  deconv_input = deconv_input/2
end  

return cnn
-- cnn = cnn:cuda()
-- cudnn.convert(cnn, cudnn)
-- cnn:evaluate()

-- input = torch.rand(1, 3, 224, 288)
-- output = cnn:forward(input:cuda())
-- print(output:size())
