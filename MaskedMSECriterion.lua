local MaskedMSECriterion, parent = torch.class('nn.MaskedMSECriterion', 'nn.Criterion')

require 'nn'

function MaskedMSECriterion:__init()
   parent.__init(self, false)
   self.mse = nn.MSECriterion()
   self.mse.sizeAverage = false
end

function MaskedMSECriterion:updateOutput(input, target)

   local mask = torch.lt(target, 0.0001)

   input[mask] = 0

   self.output = self.mse:updateOutput(input, target)
   
   self.n_valid = input:nElement() - mask:sum()
   self.output = self.output / self.n_valid
   return self.output
end

function MaskedMSECriterion:updateGradInput(input, target)

   self.mse:updateGradInput(input, target)
   self.gradInput = self.mse.gradInput / self.n_valid
   
   return self.gradInput
end

