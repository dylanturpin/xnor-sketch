--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2)) 
--  and Stavros Tsogkas (University of Toronto).
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }; options = M;

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:addTime()
    cmd:text()
    cmd:text('Torch-7 Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-cache', './cache/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', './imagenet/imagenet_raw_images/256', 'Home of ImageNet dataset, or path to sketches mat file')
    cmd:option('-dataset',  'imagenet', 'Dataset Name: imagenet |cifar')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | ccn2 | cunn')

    cmd:option('-gpu',                0,        'device ID (positive if using CUDA)')
    cmd:option('-save',               1,        'save model state every epoch')
    cmd:option('-load',               1,        'continue from last checkpoint')
    cmd:option('-tag',                '',       'additional user-tag')
    ------------- Data options ------------------------
    cmd:option('-imageSize',       256,     'Smallest side of the resized image')
    cmd:option('-cropSize',        224,     'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        250,     '#classes in the dataset')
    cmd:option('-nChannels',       1,       '#channels in the input images')
    cmd:option('-scalingFactor',   0,       '???')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-saveEveryNEpochs',      10,    'Save model every n epochs (and after final epoch)')
    cmd:option('-epochSize',       2500, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     0, 'weight decay')
    cmd:option('-shareGradInput',  true, 'Sharing the gradient memory')
    cmd:option('-binaryWeight',    false, 'Sharing the gradient memory')
    cmd:option('-testOnly',    false, 'Sharing the gradient memory')

    cmd:option('-LR',                 0.1,      '(starting) learning rate')
    cmd:option('-weightDecay',        5e-4,     'L2 penalty on the weights')
    cmd:option('-momentum',           0.9,      'momentum')
    cmd:option('-batchSize',          8,        'batch size')
    cmd:option('-seed',               123,      'torch manual random number generator seed')
    cmd:option('-numEpochs',          25,       'number of epochs to train')
    cmd:option('-weightedLoss',       0,        'asymmetric loss function based on label frequency')

    ---------- Model options ----------------------------------
    cmd:option('-netType',     'alexnet', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet | resnet')
    cmd:option('-optimType',     'sgd', 'Options: sgd | adam')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-loadParams',  'none', 'provide path to model to load the parameters')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-depth',  18, 'Depth for resnet')
    cmd:option('-shortcutType',  'B', 'type of short cut in resnet: A|B|C')
    cmd:option('-dropout', 0.5 , 'Dropout ratio')
    cmd:option('-net',                'CADC',   'network architecture')
    cmd:option('-initMethod',         'reset',  'weight initialization method')
    cmd:option('-initWeight',         0.01,     'weight initialization parameter')
    cmd:option('-initBias',           0.01,     'bias initialization parameter')


    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
        cmd:string(opt.netType, opt,
            {netType=true, retrain=true, loadParams=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    return opt
end

return M
