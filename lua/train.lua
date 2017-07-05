-- Main script for training sketch networks
dofile('parseArguments.lua')
dofile('startup.lua')

-- Load model
networks = require('networks.lua')
if opts.continue ~= '' then
    print('Loading state from file ' .. opts.continue)
    loadState(opts.continue)
else
    model = networks[opts.net](opts.nClasses)
    -- 1. Load model
    -- 2. When to initialize? Before shuffling data or after?
    -- 3. Use optim.Logger (trainLogger)?
    -- 4. Add criterion (where?)
end

if cuda then
    model:cuda()
end

-- Load data and Train
sketchDataset = require('sketchDataset.lua')
data = sketchDataset('../data/dataset_without_order_info_256.mat')
train(data)
