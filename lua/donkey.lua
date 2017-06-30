--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
paths.dofile('datasetMat.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

local loadSize   = {opt.nChannels, opt.imageSize, opt.imageSize}
local sampleSize = {opt.nChannels, opt.cropSize, opt.cropSize}


-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, input)
   opt.testMode = false

   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[3]
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(3) == oW)
   assert(out:size(2) == oH)


   -- roll> 0.45 then rotate between -5 and 5 degrees...
   if torch.uniform() > 0.45 then
      degrees = torch.random(-5,5)
      out = image.rotate(out, math.rad(degrees), 'bilinear')
   end

   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out) end

   -- mean/std
   --for i=1,opt.nChannels do -- channels
    --  if mean then out[{{i},{},{}}]:add(-mean[i]) end
   -- if std then out[{{i},{},{}}]:div(std[i]) end
   --end
   return out
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
else
   print('Creating train metadata')

   trainLoader = dataLoader{
      loadSize = loadSize,
      sampleSize = sampleSize,
      verbose = true,
      imagesFieldName = 'trainImages',
      labelsFieldName = 'trainLabels'
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
testHook = function(self, input)
   collectgarbage()
   opt.testMode = true
   local oH = sampleSize[2]
   local oW = sampleSize[3]
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
   -- mean/std
   --for i=1,opt.nChannels do -- channels
   --if mean then out[{{i},{},{}}]:add(-mean[i]) end
   --if std then out[{{i},{},{}}]:div(std[i]) end
   --end
   return out
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
else
   print('Creating test metadata')
   testLoader = dataLoader{
      loadSize = loadSize,
      sampleSize = sampleSize,
      verbose = true,
      forceClasses = trainLoader.classes, -- force consistent class indices between trainLoader and testLoader
      imagesFieldName = 'validationImages',
      labelsFieldName = 'validationLabels'
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,opt.nChannels do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,opt.nChannels do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,opt.nChannels do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,opt.nChannels do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end
