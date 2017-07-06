-- Load packages and functions.

--------------------------------------------------------------------------------
-- Load modules
--------------------------------------------------------------------------------
require "torch"
require 'paths'
require 'xlua'
require "trepl"
require "optim"
require "image"
require "nn"
require "networks.lua"

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
    if opts.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
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
    while epoch  <= opts.nEpochs do
        trainEpoch(data.train)
        test(data.val)
        
        if epoch == opts.nEpochs  or (opts.save > 0 and (epoch % opts.save == 0)) then
            print('\n==> Saving model (epoch ' .. epoch .. ')...')
            saveState(paths.concat(opts.saveDir, 'model-epoch-' .. epoch .. '.t7'))
        end
        
        epoch = epoch + 1
    end
    print('Done training for ' .. opts.numEpochs .. ' epochs!')
    print('Total time for training: '..misc.time2string(trainTimer:time().real))
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
    print('==> Optimization parameters:'); print(optimState)
    

    -- Train using batches
    batchTimer:reset()
    for t = 1, numExamples, opts.batchSize do
        local indsBatch = inds[{ {t, math.min(t+opts.batchSize-1, numExamples)} }]
        local images = data.images:index(1, indsBatch) 
        local labels = data.labels:index(1, indsBatch)
        -- Copy batch to pre-allocated space
        imageBatch:resize(images:size()):copy(images)
        labelBatch:resize(labels:size()):copy(labels)

        collectgarbage()
        if opts.binaryWeight then
            networks.meancenterConvParms(convNodes)
            networks.clampConvParms(convNodes)
            networks.realParams:copy(parameters)
            networks.binarizeConvParms(convNodes)
        end 
        -- Compute loss and gradients on current batch
        gradParameters:zero()
        local outputBatch = model:forward(imageBatch) -- BxLxDxHxW
        local lossBatch = criterion:forward(outputBatch,labelBatch)
        local gradOutput = criterion:backward(outputBatch,labelBatch)
        model:backward(imageBatch,gradOutput)
        
        if opts.binaryWeight then
            parameters:copy(realParams)
            networks.updateBinaryGradWeight(convNodes)
            if opts.optimType == 'adam' then
                gradParameters:mul(1e+9);
            end
        end        
        
        -- optimize on current mini-batch        
        local function feval(x)
            return lossBatch, gradParameters
        end
        if opts.optimType == 'sgd' then
            optim.sgd(feval, parameters, optimState)
        elseif opts.optimType == 'adam' or opts.optimType == 'adamax' then
            optim.adam(feval, parameters, optimState)
        end

        -- Compute classification error for current batch
        local top1Batch,top5Batch = computeErrors(outputBatch:float(),labels:float())
        local batch = math.ceil(t/opts.batchSize)
        lossEpoch = lossEpoch + lossBatch    
        top1Epoch = top1Epoch + top1Batch
        top5Epoch = top5Epoch + top5Batch        
        print(('Epoch (training): [%d][%d/%d]\tTime %.3f(%.3f) Loss %.4f '..
                'Top1-%%: %.2f (%.2f)  Top5-%%: %.2f (%.2f)'):format(
                epoch, batch, numBatches, epochTimer:time().real,batchTimer:time().real,
                lossBatch, top1Batch, top1Epoch/batch, top5Batch, top5Epoch/batch))
    end -- for t = 1, numExamples, opts.batchSize

    print('==> Time for epoch: ' .. misc.time2string(epochTimer:time().real)
        .. ', time per sample: ' .. misc.time2string(epochTimer:time().real/numExamples) )
end

function test(data)
    epochTimer:reset()
    local numExamples = data.images:size(1);
    local numBatches = math.ceil(numExamples/opts.batchSize)
    local inds = torch.range(1,numExamples):long()
    local loss, top1, top5 = 0,0,0
    model:evaluate()    

    -- Evaluate on batch
    batchTimer:reset()
    for t = 1, numExamples, opts.batchSize do
        local indsBatch = inds[{ {t, math.min(t+opts.batchSize-1, numExamples)} }]
        local images = data.images:index(1, indsBatch) 
        local labels = data.labels:index(1, indsBatch)
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
        local top1Batch,top5Batch = computeErrors(outputBatch:float(),labels:float())
        local batch = math.ceil(t/opts.batchSize)
        loss = loss + lossBatch    
        top1 = top1 + top1Batch
        top5 = top5 + top5Batch        
        print(('Epoch (testing): [%d][%d/%d] | Loss: %.4f | '..
                'top1: [%.2f (%.2f)] | top5: [%.2f (%.2f)]'):format(
                epoch, batch, numBatches, lossBatch, top1Batch, top1Epoch/batch,
                top5Batch, top5Epoch/batch))
    end -- for t = 1, numExamples, opts.batchSize
    
    if opts.binaryWeight then
        parameters:copy(realParams)
    end        
    
    print('==> Time for epoch: ' .. misc.time2string(epochTimer:time().real)
        .. ', time per sample: ' .. misc.time2string(epochTimer:time().real/numExamples) )
    
    
end

function loadState(modelPath)
    local state = torch.load(modelPath)
    epoch = state.epoch
    optimState = state.optimState
    model = state.model
    mode:clearState()
    if cuda then -- copy to gpu before using for computations
        model:cuda()
    end
    parameters, gradParameters = model:getParameters()
end

function saveState(modelPath)
    -- Clear intermediate and copy to CPU before saving to disk
    local model = model:clearState():clone():float()
    local state = { model = model, optimState = optimState,
        stats = stats, opts = opts, paramRegimes = paramRegimes, 
        epoch = epoch }
    torch.save(modelPath, state)
end


function computeErrors(output, target)
    -- Coputes the top1 and top5 error rate
    local batchSize = output:size(1)
    local _ , predictions = output:sort(2, true) -- descending

    -- Find which predictions match the target
    local correct = predictions:eq(
        target:long():view(batchSize, 1):expandAs(output))

    local top1 = correct:narrow(2, 1, 1):sum() / batchSize
    local top5 = correct:narrow(2, 1, 5):sum() / batchSize

    return top1 * 100, top5 * 100
end
