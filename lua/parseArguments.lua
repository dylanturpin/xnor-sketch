-- Default arguments and parsing of user-provided options
require 'torch'
require 'paths'

print('Parsing training options...')
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
cmd:option('-augment',         false,     'random cropping, rotation, flipping')
-- Training options --------------------
cmd:option('-nEpochs',          300,    'Number of total epochs to run')
cmd:option('-batchSize',        100,   'mini-batch size (1 = pure stochastic)')
cmd:option('-LR',               0.01, 'starting learning rate; see recipe defined in startup.lua')
cmd:option('-weightDecay',      5e-4,     'L2 penalty on the weights')
cmd:option('-momentum',         0.9,      'momentum')
cmd:option('-seed',             123,      'torch manual random number generator seed')
-- Model options ----------------------------------
cmd:option('-netType',          'sketchnet', 'Options: alexnetxnor | sketchnetxnor| alexnetowtbn | sketchanet')
cmd:option('-optimType',        'sgd',      'Options: sgd | adam')
cmd:option('-initMethod',       'xavier',    'weight initialization method')
-- Miscellaneous (device and storing options) ---------------------------------
cmd:text('===> Miscellaneous options')
cmd:option('-gpu',                1,        'device ID (positive if using CUDA)')
cmd:option('-save',               10,       'Save model every n epochs (and after final epoch)')
cmd:option('-continue',           '',       'load state from file and continue training')
cmd:option('-debug',              false,     'execute debugging commands')
cmd:option('-tag',                '',       'additional user-tag')

-- Parse arguments
opts = cmd:parse(arg or {})

------------------------------------------------------------------------------------
-- Setup log file
------------------------------------------------------------------------------------
opts.saveDir = cmd:string('../output/models/torch', opts, 
    {dir=true,gpu=true,save=true,continue=true,debug=true})
paths.mkdir(opts.saveDir)
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
