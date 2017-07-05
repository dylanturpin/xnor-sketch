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

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------------------
-- Define training and testing functions
--------------------------------------------------------------------------------

-- Training code ---------------------------------------------------------------
function train(data)

    -- Initialize optimState if it has not been loaded from disk
    if not optimState then
        optimState = {  learningRate = opts.LR,
                        learningRateDecay = 0.0,
                        momentum = opts.momentum,
                        dampening = 0.0,
                        weightDecay = opts.weightDecay
                    }
    end
 
    -- By default we follow a known recipe for a 55-epoch training unless the
    -- user has manually provided a learning rate.
    local function paramsForEpoch(epoch)
        -- Returns:
        --    diff to apply to optimState,
        --    true IFF this is the first epoch of a new regime
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
    
    -- WARNING: This command goes AFTER transfering the network to the GPU!
    -- Retrieve parameters and gradients:
    -- extracts and flattens all the trainable parameters of the model into a vector
    -- Parameters are references: when model:backward is called, these are changed
    parameters,gradParameters = model:getParameters()
    realParams = parameters:clone()
    if cudann then
        convNodes = model:findModules('cudnn.SpatialConvolution')
    else
        convNodes = model:findModules('nn.SpatialConvolution')
    end
    
    
    -- Get data for training and validation
    local trainData = data:subset('train')
    local valData   = data:subset('val')
    
    local trainTimer = torch.Timer()
    while epoch  <= opts.nEpochs do
        runEpoch('train', trainData)
        runEpoch('val',   valData)

        if (opts.save > 0 and (epoch % opts.save > 0)) or epoch == opts.nEpochs then
            print('\n==> Saving model (epoch ' .. epoch .. ')...')
            saveState(paths.concat(opts.saveDir, 'model-epoch-' .. epoch .. '.t7'))
        end
    end
    print('Done training for ' .. opts.numEpochs .. ' epochs!')
    print('Total time for training: '..misc.time2string(trainTimer:time().real))
end


-- Execute a single epoch of training (or validation) -------------------------
function runEpoch(mode, data)
    
    local numExamples = data.images:size(1);
    local numBatches = math.ceil(numExamples/opts.batchSize)
    local lossEpoch = 0, top1Sum = 0, top5Sum = 0

    print('\n--------------- ' .. mode .. ' epoch ' .. epoch .. '/' .. opts.numEpochs
        ..' [#examples = ' .. numExamples .. ', batchSize = ' .. opts.batchSize
        .. ', #batches: ' .. numBatches .. '] ----------------')

    local inTrainMode = mode == 'train'
    if inTrainMode then 
        local params, newRegime = paramsForEpoch(epoch)
        if newRegime then
            optimState.learningRate = params.learningRate
            optimState.weightDecay = params.weightDecay
            if (opts.optimType == "adam") or (opts.optimType == "adamax") then
                optimState.learningRate = optimState.learningRate*0.1
            end
        end
        print('Optimization parameters:'); print(optimState)
        model:training()
        data:shuffle()
    else 
        model:evaluate() 
    end

    -- Train using batches
    epochTimer = epochTimer or torch.Timer(); epochTimer:reset()
    for t = 1, numExamples, opts.batchSize do
        local imageBatch, labelBatch = data:batch(t, math.min(t+opts.batchSize-1, numExamples))
        
        -- closure to evaluate f(X) and df/dX (L and dL/dw)
        collectgarbage()
        local lossBatch = 0
        local outputBatch = model:forward(imageBatch) -- BxLxDxHxW
        local function feval(x)
            gradParameters:zero()
            local gradOutput
            lossBatch, gradOutput = networks.spatialCrossEntropy(outputBatch,labelBatch,weights)
            model:backward(imageBatch,gradOutput)
            return lossBatch, gradParameters
        end
        if inTrainMode then -- optimize on current mini-batch
            optim.sgd(feval, parameters, optimState)
        end

        -- Print stats for current batch
        local maxScores,prediction = outputBatch:max(2) -- max score over labels  (B1DHW)
        prediction, labelBatch = prediction:float():view(-1), labelBatch:float():view(-1)
        confusion:batchAdd(prediction,labelBatch) --print(confusion)
        local s, batch = computeErrorStats(prediction,labelBatch), math.ceil(t/opts.batchSize)
        lossEpoch = lossEpoch + lossBatch
    end -- for t = 1, numExamples, opts.batchSize

    -- print and reset confusion matrix and timings
    print(''); print(confusion); confusion:zero() print('')
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


function computeScore(output, target, nCrops)
    if nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
        --:exp()
        :sum(2):squeeze(2)
    end

    -- Coputes the top1 and top5 error rate
    local batchSize = output:size(1)

    local _ , predictions = output:float():sort(2, true) -- descending

    -- Find which predictions match the target
    local correct = predictions:eq(
        target:long():view(batchSize, 1):expandAs(output))

    local top1 = correct:narrow(2, 1, 1):sum() / batchSize
    local top5 = correct:narrow(2, 1, 5):sum() / batchSize

    return top1 * 100, top5 * 100
end
