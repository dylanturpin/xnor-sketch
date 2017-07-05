--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--debugger = require('fb.debugger')

require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

local options = paths.dofile('opts.lua')
opt = options.parse(arg)

-- Use 'float' as the default data type
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(options.seed)


-- CUDA?
if opt.gpu > 0 then
    cuda = true
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed)
end

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')

opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

loaded = mattorch.load(opt.data)
loaded.trainImages = 255 - loaded.trainImages
loaded.validationImages = 255 - loaded.validationImages

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')


epoch = opt.epochNumber
if opt.testOnly then
	test()
else
  for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
  end
end
