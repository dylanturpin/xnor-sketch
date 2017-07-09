-- Main script for training sketch networks
-- TODO: Use optim.Logger (trainLogger)?
-- TODO: Make sure saveState() and loadState() work properly
-- TODO: Should I also load opts in loadState()?
-- TODO: change last layer of network to nn.Linear
dofile('startup.lua')

-- Load state (model, optimState, epoch) from user-specified file 
-- or initialize state and model.
if opts.continue ~= '' then
    print('Loading state from file ' .. opts.continue)
    loadState(opts.continue)
else
    model = networks[opts.netType]()
    model:apply(networks.initialize(opts.initMethod))
    -- Initialize optimState if it has not been loaded from disk
    optimState = {  learningRate = opts.LR,
        learningRateDecay = 0.0,
        momentum = opts.momentum,
        dampening = 0.0,
        weightDecay = opts.weightDecay
    }
    epoch = 1
end

-- Define criterion and push criterion and model to the GPU
criterion = nn.ClassNLLCriterion()
if cuda then
    print('Moving model and criterion to GPU')
    model:cuda()
    criterion:cuda()
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

-- Load data and Train
local data = SketchDataset('../data/dataset_without_order_info_256.mat')
train(data)

