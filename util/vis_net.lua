require 'nn'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

generateGraph = require 'optnet.graphgen'

graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  return oldData .. '\n' .. 'Size: '.. tensor:numel()
end
}

net = dofile('model_novgg.lua')
net:cuda()

cudnn.convert(net, cudnn)

input = torch.rand(1,3,200,280)
g = generateGraph(net, input:cuda(), graphOpts)

local modelname = 'novgg'
require 'graph'
graph.dot(g,modelname,modelname)
