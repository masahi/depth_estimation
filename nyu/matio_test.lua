local matio = require 'matio'

local file = io.open("file_names_shuffled.txt")
local file_names = {}
local count = 1
for line in file:lines() do
   file_names[count] = line
   count = count + 1
end        
local npy4th = require 'npy4th'

for i = 1, #file_names do
   --print(i)
  -- local a = torch.Tensor(10000):zero()
  -- local data = matio.load(file_names[i])
   local f = file_names[i] .. '\n_rgb.npy'
   local rgb = npy4th.loadnpy(f)
   local depth = npy4th.loadnpy(file_names[i] .. '\n_depth.npy')   
--   collectgarbage()
end   

