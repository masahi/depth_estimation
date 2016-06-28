npy4th = require 'npy4th'
require 'image'

depths = npy4th.loadnpy('pred.npy')
itorch.image(depths)
