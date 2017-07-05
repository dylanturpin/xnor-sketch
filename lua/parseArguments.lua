-- Default arguments and parsing of user-provided options
cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a network for semantic segmentation')
cmd:text()
cmd:text('===> Default options:')
-- Data options (#images, preprocessing etc.) ---------------------------------
cmd:option('-imageSize',       256,     'Smallest side of the resized image')
cmd:option('-cropSize',        224,     'Height and Width of image crop to be used as input layer')
cmd:option('-nClasses',        250,     '#classes in the dataset')
cmd:option('-nChannels',       1,       '#channels in the input images')
cmd:option('-scalingFactor',   0,       '???')
-- Training options --------------------
cmd:option('-nEpochs',          55,    'Number of total epochs to run')
cmd:option('-batchSize',        128,   'mini-batch size (1 = pure stochastic)')
cmd:option('-LR',               0.0, 'learning rate; if set, overrides default LR/WD recipe')
cmd:option('-weightDecay',      5e-4,     'L2 penalty on the weights')
cmd:option('-momentum',         0.9,      'momentum')
cmd:option('-seed',             123,      'torch manual random number generator seed')
cmd:option('-shareGradInput',   true, 'Sharing the gradient memory')
cmd:option('-binaryWeight',     false, 'Sharing the gradient memory')
cmd:option('-testOnly',         false, 'Sharing the gradient memory')
-- Model options ----------------------------------
cmd:option('-netType',     'alexnet', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet | resnet')
cmd:option('-optimType',     'sgd', 'Options: sgd | adam')
cmd:option('-retrain',     'none', 'provide path to model to retrain with')
cmd:option('-loadParams',  'none', 'provide path to model to load the parameters')
cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
cmd:option('-dropout', 0.5 , 'Dropout ratio')
cmd:option('-net',                'CADC',   'network architecture')
cmd:option('-initMethod',         'reset',  'weight initialization method')
cmd:option('-initWeight',         0.01,     'weight initialization parameter')
cmd:option('-initBias',           0.01,     'bias initialization parameter')
-- Miscellaneous (device and storing options) ---------------------------------
cmd:text('===> Miscellaneous options')
cmd:option('-gpu',                0,        'device ID (positive if using CUDA)')
cmd:option('-save',               10,       'Save model every n epochs (and after final epoch)')
cmd:option('-continue',           '',       'load state from file and continue training')
cmd:option('-tag',                '',       'additional user-tag')

-- Parse arguments
opts = cmd:parse(arg or {})

------------------------------------------------------------------------------------
-- Setup log file
------------------------------------------------------------------------------------
-- We temporarily remove options that are not used to form the saveDir name.
-- We do the same thing for the the class name, so that it is explicitly placed
-- at the beginning of the directory name.
local gpu    = opts.gpu;   opts.gpu   = nil
local save   = opts.save;  opts.save  = nil
opts.saveDir = cmd:string('../output/models', opts, {dir=true})
paths.mkdir(opts.saveDir)
opts.gpu = gpu; opts.save = save; 
cmd:log(opts.saveDir .. '/log-' .. os.date('%d-%m-%Y-%X') , opts)

------------------------------------------------------------------------------------
-- Set device and random seed. We include these here to keep startup.lua standalone
------------------------------------------------------------------------------------
torch.manualSeed(opts.seed)

-- CUDA?
if opts.gpu > 0 then
    cuda = true
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opts.gpu)
    cutorch.manualSeed(opts.seed)
end
