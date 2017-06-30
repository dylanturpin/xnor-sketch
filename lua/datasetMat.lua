require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')
os.execute('mkdir ' .. opt.cache .. '/tmp');
local initcheck = argcheck{
   pack=false,
   help=[[
     A dataset class for images in matlab (.mat) datafile.
]],

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},

   {name="imagesFieldName",
    type="string",
    help="name of the images field name in the loaded mat file",
    opt = true},

   {name="labelsFieldName",
    type="string",
    help="name of the labels field name in the loaded mat file",
    opt = true}
}

function dataset:__init(...)
   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   if not self.loadSize then self.loadSize = self.sampleSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return loaded[self.labelsFieldName]:size()[1]
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable

   assert(dataTable[1]:dim() == 3)

   data = torch.Tensor(quantity,
		       self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   scalarLabels = torch.LongTensor(quantity):fill(-1111)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      scalarLabels[i] = scalarTable[i]
   end
   return data, scalarLabels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      -- get a random image, convert from byte tensort to float tensor
      local randomIndex = torch.random(1, loaded[self.imagesFieldName]:size(1))
      local out = loaded[self.imagesFieldName][{randomIndex,{},{},{}}]:float()
      out = self:sampleHookTrain(out)

      table.insert(dataTable, out)
      table.insert(scalarTable, loaded[self.labelsFieldName][randomIndex])
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end

function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   for i=1,quantity do
      -- load the sample
      local out = loaded[self.imagesFieldName][{indices[i],{},{},{}}]:float()
      out = self:sampleHookTest(out)

      table.insert(dataTable, out)
      table.insert(scalarTable, loaded[self.labelsFieldName][indices[i]])
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end

return dataset
