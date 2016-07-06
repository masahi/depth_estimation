require 'loadcaffe'
require 'nn'

local proto_file = 'vgg/VGG_ILSVRC_16_layers_deploy.prototxt'
local model_file = 'vgg/VGG_ILSVRC_16_layers.caffemodel'

local vgg_net = loadcaffe.load(proto_file, model_file)

local cnn = nn.Sequential()

local n_output = -1

for i = 1, #vgg_net do
  local layer = vgg_net:get(i)
  if torch.isTypeOf(layer, 'nn.SpatialConvolution') then
    if i == 1 then
      local w = layer.weight:clone()
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])

    end
    
    cnn:add(layer)
    cnn:add(nn.SpatialBatchNormalization(layer.nOutputPlane,1e-3))
    cnn:add(nn.ReLU(true))

    n_output = layer.nOutputPlane
    
  end

  if torch.isTypeOf(layer, 'nn.SpatialMaxPooling') then
     cnn:add(layer)
  end
  
end

local n_deconv_input = n_output

cnn:add(nn.SpatialFullConvolution(n_deconv_input, n_deconv_input / 2, 4, 4, 2, 2, 1, 1))
--cnn:add(nn.SpatialConvolution
cnn:add(nn.SpatialBatchNormalization(n_deconv_input / 2)):add(nn.ReLU(true))

cnn:add(nn.SpatialFullConvolution(n_deconv_input / 2 , n_deconv_input / 4, 4, 4, 2, 2, 1, 1))
cnn:add(nn.SpatialBatchNormalization(n_deconv_input / 4)):add(nn.ReLU(true))

cnn:add(nn.SpatialFullConvolution(n_deconv_input / 4 , n_deconv_input / 8, 4, 4, 2, 2, 1, 1))
cnn:add(nn.SpatialBatchNormalization(n_deconv_input / 8)):add(nn.ReLU(true))

cnn:add(nn.SpatialFullConvolution(n_deconv_input / 8 , n_deconv_input / 16, 4, 4, 2, 2, 1, 1))
cnn:add(nn.SpatialBatchNormalization(n_deconv_input / 16)):add(nn.ReLU(true))

cnn:add(nn.SpatialFullConvolution(n_deconv_input / 16 , 1, 4, 4, 2, 2, 1, 1))

return cnn
