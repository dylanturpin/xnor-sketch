-- Default arguments and parsing of user-provided options
require 'torch'
require 'paths'

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a network for semantic segmentation')
cmd:text()
cmd:text('===> Default options:')
-- Data options (#images, preprocessing etc.) ---------------------------------
cmd:option('-imageSize',       256,     'Smallest side of the resized image')
cmd:option('-cropSize',        225,     'Height and Width of image crop to be used as input layer')
cmd:option('-nClasses',        250,     '#classes in the dataset')
cmd:option('-nChannels',       1,       '#channels in the input images')
-- Training options --------------------
cmd:option('-nEpochs',          300,    'Number of total epochs to run')
cmd:option('-batchSize',        32,   'mini-batch size (1 = pure stochastic)')
cmd:option('-LR',               0.0001, 'starting learning rate; see recipe defined in startup.lua')
cmd:option('-weightDecay',      5e-4,     'L2 penalty on the weights')
cmd:option('-momentum',         0.9,      'momentum')
cmd:option('-seed',             123,      'torch manual random number generator seed')
-- Model options ----------------------------------
cmd:option('-netType',          'sketchnetxnor', 'Options: alexnetxnor | sketchnetxnor| alexnetowtbn | sketchanet')
cmd:option('-optimType',        'sgd',      'Options: sgd | adam')
cmd:option('-initMethod',       'xavier',    'weight initialization method')
-- Miscellaneous (device and storing options) ---------------------------------
cmd:text('===> Miscellaneous options')
cmd:option('-gpu',                0,        'device ID (positive if using CUDA)')
cmd:option('-save',               10,       'Save model every n epochs (and after final epoch)')
cmd:option('-continue',           '',       'load state from file and continue training')
cmd:option('-debug',              true,     'execute debugging commands')
cmd:option('-tag',                '',       'additional user-tag')

-- Parse arguments
opts = cmd:parse(arg or {})

-- Make sure binary networks use binary weights
opts.binaryWeight = opts.netType == 'sketchnetxnor' or opts.netType == 'alexnetxnor'

------------------------------------------------------------------------------------
-- Setup log file
------------------------------------------------------------------------------------
-- We temporarily remove options that are not used to form the saveDir name.
-- We do the same thing for the the class name, so that it is explicitly placed
-- at the beginning of the directory name.
local gpu       = opts.gpu;   opts.gpu   = nil
local save      = opts.save;  opts.save  = nil
local continue  = opts.continue; opts.continue = nil
local debug     = opts.debug; opts.debug = nil
opts.saveDir = cmd:string('../output/models', opts, {dir=true})
paths.mkdir(opts.saveDir)
opts.gpu = gpu; opts.save = save; opts.continue = continue; opts.debug = debug
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
