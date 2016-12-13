require 'nn'

local cnn = nn.Sequential()

n_blocks = 5
n_inputs = torch.LongTensor({3, 48, 128,
                             128, 256, 256,
                             256, 256, 512,
                             512, 512, 512,
                             512, 512, 512
                             })

n_outputs = torch.LongTensor({48, 128, 128,
                              256, 256, 256,
                              256, 512, 512,
                              512, 512, 512,
                              512, 512, 512                              
                             })

in_idx = 1
out_idx = 1

for i = 1,n_blocks do
   cnn:add(nn.SpatialConvolution(n_inputs[in_idx], n_outputs[out_idx], 5, 5, 2, 2 ,2, 2))
   cnn:add(nn.SpatialBatchNormalization(n_outputs[out_idx])):add(nn.ReLU(true))
   
   cnn:add(nn.SpatialConvolution(n_inputs[in_idx+1], n_outputs[out_idx+1], 3, 3, 1,1,1,1))
   cnn:add(nn.SpatialBatchNormalization(n_outputs[out_idx+1])):add(nn.ReLU(true))
   
   cnn:add(nn.SpatialConvolution(n_inputs[in_idx+2], n_outputs[out_idx+2], 3, 3, 1,1,1,1))
   cnn:add(nn.SpatialBatchNormalization(n_outputs[out_idx+2])):add(nn.ReLU(true))

   in_idx = in_idx + 3
   out_idx = out_idx + 3
end

n_inputs = torch.LongTensor({
                             512, 512, 512,
                             512, 512, 512 ,                             
                             512, 512, 256,
                             256, 256, 256,
                             128, 128, 48
                             })

n_outputs = torch.LongTensor({
                              512, 512, 512,
                              512, 512, 512,                             
                              512, 256, 256,
                              256, 256, 128,
                              128, 48, 1
                             })

in_idx = 1
out_idx = 1

for i = 1,n_blocks do
   cnn:add(nn.SpatialFullConvolution(n_inputs[in_idx], n_outputs[out_idx], 4, 4, 2, 2 ,1, 1))
   cnn:add(nn.SpatialBatchNormalization(n_outputs[out_idx])):add(nn.ReLU(true))
   
   cnn:add(nn.SpatialConvolution(n_inputs[in_idx+1], n_outputs[out_idx+1], 3, 3, 1,1,1,1))
   cnn:add(nn.SpatialBatchNormalization(n_outputs[out_idx+1])):add(nn.ReLU(true))
   
   cnn:add(nn.SpatialConvolution(n_inputs[in_idx+2], n_outputs[out_idx+2], 3, 3, 1,1,1,1))

   if i < n_blocks then
   cnn:add(nn.SpatialBatchNormalization(n_outputs[out_idx+2])):add(nn.ReLU(true))
   end
   
   in_idx = in_idx + 3
   out_idx = out_idx + 3
   
end

return cnn
