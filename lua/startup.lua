-- Load packages and functions.
--------------------------------------------------------------------------------
-- Load packages
-------------------------------------------------------------------------------
print('Loading packages...')
require "torch"
require 'paths'
require 'xlua'
require "trepl"
require "optim"
require "image"
require "nn"
require "SketchDataset" -- data loading class
require "networks"      -- contains network definitions
local debugger = require "fb.debugger"

dofile('parseArguments.lua')

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------------------
-- Training and testing functions
--------------------------------------------------------------------------------

-- Define scheme for learning rate, weight decay etc. for each epoch.
-- By default we follow a known recipe for a 55-epoch training unless 
-- the user has manually provided a learning rate.
-- Returns:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if not opts.LR  then -- if manually specified
        return { }
    end
    local LR = opts.LR
    local WD = opts.weightDecay
    local reg = torch.range(1,opts.nEpochs,math.ceil(opts.nEpochs/6))
    local regimes = {        
        
        -- start, end,              LR,     WD,
        {  1,     reg[2]-1,         LR,     WD },
        { reg[2], reg[3]-1,         LR/2,   WD },
        { reg[3], reg[4]-1,         LR/10,  0  },
        { reg[4], reg[5]-1,         LR/20,  0  },
        { reg[5], opts.nEpochs,     LR/100, 0  },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

function train(data)     
    data:split() -- Split data into training and validation sets.
    
    -- Run a round of training and validation for each epoch.
    local trainTimer = torch.Timer()
    while epoch <= opts.nEpochs do
        trainEpoch(data.train)
        test(data.val)
        
        if epoch == opts.nEpochs  or (opts.save > 0 and (epoch % opts.save == 0)) then
            print('\n==> Saving model (epoch ' .. epoch .. ')...\n')
            saveState(paths.concat(opts.saveDir, 'model-epoch-' .. epoch .. '.t7'))
        end
        
        epoch = epoch + 1
    end
    print('\nDone training for ' .. opts.nEpochs .. ' epochs!')
    print('Total time for training: '..trainTimer:time().real)
end


-- Preallocate memory for batches
local imageBatch, labelBatch
if cuda then
    imageBatch = torch.CudaTensor()
    labelBatch = torch.CudaTensor()
else
    imageBatch = torch.FloatTensor()
    labelBatch = torch.FloatTensor()
end

-- Timers for epochs and batches
local epochTimer = torch.Timer()
local batchTimer = torch.Timer()

-- Execute a single epoch of training (or validation) -------------------------
function trainEpoch(data)
    print('==> Starting training for epoch '..epoch)
    epochTimer:reset()
    local numExamples = data.images:size(1);
    local numBatches = math.ceil(numExamples/opts.batchSize)
    local lossEpoch, top1Epoch, top5Epoch = 0,0,0
    local inds = torch.randperm(numExamples):long() -- used to shuffle data
    model:training()
    
    local params, newRegime = paramsForEpoch(epoch)
    if newRegime then
        optimState.learningRate = params.learningRate
        optimState.weightDecay = params.weightDecay
        if (opts.optimType == "adam") or (opts.optimType == "adamax") then
            optimState.learningRate = optimState.learningRate*0.1
        end
    end
    print('Optimization parameters:'); 
    print('learningRate: ' .. optimState.learningRate)
    print('weightDecay: ' .. optimState.weightDecay)
    print('learningRateDecay: ' .. optimState.learningRateDecay)
    print('momentum: ' .. optimState.momentum)
    print('dampening: ' .. optimState.dampening)
    
    -- Train using batches
    for b = 1, numExamples, opts.batchSize do
        batchTimer:reset()
        local numExamplesSoFar = math.min(b+opts.batchSize-1, numExamples)
        local indsBatch = inds[{ {b, numExamplesSoFar} }]
        local images,labels = getBatch(data,indsBatch,opts.augment)
        -- Copy batch to pre-allocated space
        imageBatch:resize(images:size()):copy(images)
        labelBatch:resize(labels:size()):copy(labels)
--        debugger.enter()
                
        collectgarbage()
        if opts.binaryWeight then
            networks.meancenterConvParms(convNodes)
            networks.clampConvParms(convNodes)
            realParams:copy(parameters)
            networks.binarizeConvParms(convNodes)
        end 
        -- Compute loss and gradients on current batch
        gradParameters:zero()
        local outputBatch = model:forward(imageBatch) -- BxL
        local lossBatch   = criterion:forward(outputBatch,labelBatch)
        local gradOutput  = criterion:backward(outputBatch,labelBatch)
        model:backward(imageBatch,gradOutput)
        
        if opts.binaryWeight then
            parameters:copy(realParams)
            networks.updateBinaryGradWeight(convNodes)
            if opts.optimType == 'adam' then
                gradParameters:mul(1e+9);
            end
        end        
        
        -- optimize on current mini-batch        
        local function feval()
            return lossBatch, gradParameters
        end
        if opts.optimType == 'sgd' then
            optim.sgd(feval, parameters, optimState)
        elseif opts.optimType == 'adam' or opts.optimType == 'adamax' then
            optim.adam(feval, parameters, optimState)
        end

        -- Compute classification error for current batch
        local batchSize = imageBatch:size(1)
        outputBatch = outputBatch:view(batchSize,-1):float()
        local top1Batch,top5Batch = computeAccuracy(outputBatch,labels)
        local batch = math.ceil(b/opts.batchSize)
        lossEpoch = lossEpoch + lossBatch    
        top1Epoch = top1Epoch + top1Batch
        top5Epoch = top5Epoch + top5Batch        
        print(('Training Epoch: [%d][%d/%d]\t %.3fs Loss %.4f '..
                'Top1-%%: %.2f (%.2f)  Top5-%%: %.2f (%.2f)'):format(
                epoch, batch, numBatches, batchTimer:time().real,lossBatch, 
                top1Batch/batchSize*100, top1Epoch/numExamplesSoFar*100, 
                top5Batch/batchSize*100, top5Epoch/numExamplesSoFar*100))
    end -- for t = 1, numExamples, opts.batchSize

    print('Time for epoch: ' .. epochTimer:time().real
        .. ', time per sample: ' .. epochTimer:time().real/numExamples..'\n')
end

function test(data)
    print('==> Starting testing on validation data for epoch '..epoch)
    epochTimer:reset()
    local numExamples = data.images:size(1);
    local numBatches = math.ceil(numExamples/opts.batchSize)
    local inds = torch.range(1,numExamples):long()
    local loss, top1, top5 = 0,0,0
    model:evaluate()    

    -- Evaluate on batch
    for b = 1, numExamples, opts.batchSize do
        batchTimer:reset()
        local numExamplesSoFar = math.min(b+opts.batchSize-1, numExamples)
        local indsBatch = inds[{ {b, numExamplesSoFar} }]
        local images,labels = getBatch(data,indsBatch,false)
        
        -- Copy batch to pre-allocated space
        imageBatch:resize(images:size()):copy(images)
        labelBatch:resize(labels:size()):copy(labels)

        collectgarbage()
        if opts.binaryWeight then
            networks.binarizeConvParms(convNodes)
        end 
        -- Compute loss and gradients on current batch
        local outputBatch = model:forward(imageBatch) -- BxLxDxHxW
        local lossBatch = criterion:forward(outputBatch,labelBatch)
                
        -- Compute classification error for current batch
        local batchSize = images:size(1)
        outputBatch = outputBatch:view(batchSize,-1):float()
        local top1Batch,top5Batch = computeAccuracy(outputBatch,labels)
        local batch = math.ceil(b/opts.batchSize)
        loss = loss + lossBatch    
        top1 = top1 + top1Batch
        top5 = top5 + top5Batch        
        print(('Testing Epoch: [%d][%d/%d]\t%.3fs Loss %.4f '..
                'Top1-%%: %.2f (%.2f)  Top5-%%: %.2f (%.2f)'):format(
                epoch, batch, numBatches, batchTimer:time().real, lossBatch, 
                top1Batch/batchSize*100, top1/numExamplesSoFar*100, 
                top5Batch/batchSize*100, top5/numExamplesSoFar*100))
    end -- for t = 1, numExamples, opts.batchSize
    
    if opts.binaryWeight then
        parameters:copy(realParams)
    end        
    
    print('Time for epoch: ' .. epochTimer:time().real
        .. ', time per sample: ' .. epochTimer:time().real/numExamples..'\n')
    
    
end

function getBatch(data,inds,augment)
    local batchSize = inds:nElement()
    local labels = data.labels:index(1, inds)
    
    -- Remember that jittering must take place in the CPU
    local images
    if augment then
        local iW,iH = opts.imageSize, opts.imageSize
        local oW,oH = opts.cropSize, opts.cropSize
        images = torch.FloatTensor(batchSize,opts.nChannels,oH,oW)
        for i=1,batchSize do
            -- do random crop
            local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
            local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
            images[i] = image.crop(data.images[i], w1, h1, w1 + oW, h1 + oH);
            -- roll> 0.45 then rotate between -5 and 5 degrees...
            if torch.uniform() > 0.45 then
                local degrees = torch.random(-5,5)
                images[i] = image.rotate(images[i], math.rad(degrees));
            end
            -- do hflip with probability 0.5
            if torch.uniform() > 0.5 then 
                image.hflip(images[i],images[i]); 
            end
        end
    else
        -- Even with no augmentation we still have to (center-) crop images
        local pad = math.ceil((opts.imageSize-opts.cropSize)/2)
        images = data.images:index(1, inds)
        images = images:narrow(3,pad,opts.cropSize):narrow(4,pad,opts.cropSize)
    end
    
    -- Make sure images and labels are within acceptable range
    if opts.debug then
        assert(images:min()>=0 and images:max()<=1)
        assert(labels:min()>=1 and labels:max()<=opts.nClasses)
    end
        
    return images,labels
end

function computeAccuracy(output, target)
    -- Coputes the top1 and top5 error rate
    local batchSize = output:size(1)
    local _ , predictions = output:sort(2, true) -- descending (BxC tensor)

    -- Find which predictions match the target
    local correct = predictions:eq(target:long():view(batchSize,1):expandAs(output))

    local top1 = correct:narrow(2, 1, 1):sum() 
    local top5 = correct:narrow(2, 1, 5):sum()

    return top1, top5
end

function loadState(modelPath)
    local state = torch.load(modelPath)
    epoch = state.epoch
    optimState = state.optimState
    model = state.model
    if cuda then -- copy to gpu before using for computations
        model:cuda()
    end
    parameters, gradParameters = model:getParameters()
end

function saveState(modelPath)
    -- Clear intermediate and copy to CPU before saving to disk
    model = model:clearState():float()
    local state = { model = model, 
                    optimState = optimState, 
                    opts = opts, 
                    epoch = epoch 
                  }
    torch.save(modelPath, state)
    -- Move back to gpu
    if cuda then
        model:cuda()
    end
    collectgarbage()
end
