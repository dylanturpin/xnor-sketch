-- Network architectures
local P = {}; networks = P;
require "nn"

local nClasses = opts.nClasses

-- Commonly used layers - allow easy switch between CPU and GPU
local Convolution,MaxPooling,ReLU,CrossMapLRN,Dropout,BatchNormalization
if cudnn then
    Convolution = cudnn.SpatialConvolution
    MaxPooling  = cudnn.SpatialMaxPooling
    CrossMapLRN = cudnn.SpatialCrossMapLRN
    BatchNorm   = cudnn.SpatialBatchNormalization
    ReLU        = cudnn.ReLU
else
    Convolution = nn.SpatialConvolution
    MaxPooling  = nn.SpatialMaxPooling
    CrossMapLRN = nn.SpatialCrossMapLRN
    BatchNorm   = nn.SpatialBatchNormalization
    ReLU        = nn.ReLU    
end
Dropout = nn.Dropout

-- New layers 
-- New class for binary activations
local BinActiveZ , parent= torch.class('nn.BinActiveZ', 'nn.Module')
function BinActiveZ:updateOutput(input)
    local s = input:size()
    self.output:resizeAs(input):copy(input)
    self.output=self.output:sign();
    return self.output
end

function BinActiveZ:updateGradInput(input, gradOutput)
    local s = input:size()
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput[input:ge(1)]=0
    self.gradInput[input:le(-1)]=0
    return self.gradInput
end

-- Binary activation layer
local function Activation()
    local C= nn.Sequential()
    C:add(nn.BinActiveZ())
    return C
end

-- Binary convolution
local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    local C= nn.Sequential()
    C:add(BatchNorm(nInputPlane,1e-4,false))
    C:add(Activation())
    C:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
    return C
end

-- Binary convolution + max pooling
local function BinMaxConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,mW,mH)
    local C= nn.Sequential()
    C:add(BatchNorm(nInputPlane,1e-4,false))
    C:add(Activation())
    C:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
    C:add(MaxPooling(3,3,2,2))
    return C
end


-- Network architectures-------------------------------------------------
-- Alexnet model ---------------------------------------------------------
function P.alexnet()
    require 'cudnn'
    local function ContConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local C= nn.Sequential()
        C:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
        C:add(BatchNorm(nOutputPlane,1e-3))
        C:add(ReLU(true))
        return C
    end

    local net = nn.Sequential()
    net:add(ContConvolution(3,96,11,11,4,4,2,2))       -- 224 -> 55
    net:add(MaxPooling(3,3,2,2))                   -- 55 ->  27
    net:add(ContConvolution(96,256,5,5,1,1,2,2))       --  27 -> 27  
    net:add(MaxPooling(3,3,2,2))                     --  27 ->  13
    net:add(ContConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
    net:add(ContConvolution(384,384,3,3,1,1,1,1)) 
    net:add(ContConvolution(384,256,3,3,1,1,1,1)) 
    net:add(MaxPooling(3,3,2,2))           
    net:add(nn.SpatialDropout(opts.dropout))
    net:add(ContConvolution(256,4096,6,6))
    net:add(nn.SpatialDropout(opts.dropout))           
    net:add(ContConvolution(4096,4096,1,1)) 
    net:add(Convolution(4096, nClasses,1,1))
    net:add(nn.View(nClasses))
    net:add(nn.LogSoftMax())

    return net

end

-- Alexnet  + xnor -----------------------------------------
function P.alexnetxnor()

    local net = nn.Sequential()

    net:add(Convolution(3,96,11,11,4,4,2,2))      
    net:add(BatchNorm(96,1e-5,false))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3,2,2))

    net:add(BinMaxConvolution(96,256,5,5,1,1,2,2))        
    net:add(BinConvolution(256,384,3,3,1,1,1,1))      
    net:add(BinConvolution(384,384,3,3,1,1,1,1)) 
    net:add(BinMaxConvolution(384,256,3,3,1,1,1,1))        
    net:add(BinConvolution(256,4096,6,6))        
    net:add(BinConvolution(4096,4096,1,1))

    net:add(BatchNorm(4096,1e-3,false))
    net:add(ReLU(true))
    net:add(Convolution(4096, nClasses,1,1))

    net:add(nn.View(nClasses))
    net:add(nn.LogSoftMax())

    return net
end

-- Sketch-A-Net ----------------------------------------
function P.sketchanet()

    local net = nn.Sequential()

    -- index 0 is input
    -- index 1
    net:add(Convolution(opts.nChannels,64,15,15,3,3,0,0))      
    --net:add(BatchNorm(64,1e-5,false))
    -- index 2
    net:add(ReLU(true))
    -- index 3
    net:add(MaxPooling(3,3,2,2))

    -- index 4-6
    net:add(Convolution(64,128,5,5,1,1,0,0))      
    --net:add(BatchNorm(128,1e-5,false))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3,2,2))

    -- index 7-8
    net:add(Convolution(128,256,3,3,1,1,1,1))      
    --net:add(BatchNorm(256,1e-5,false))
    net:add(ReLU(true))

    -- index 9-10
    net:add(Convolution(256,256,3,3,1,1,1,1))      
    --net:add(BatchNorm(256,1e-5,false))
    net:add(ReLU(true))

    -- index 11-13
    net:add(Convolution(256,256,3,3,1,1,1,1))      
    --net:add(BatchNorm(256,1e-5,false))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3,2,2))

    -- index 14-16 FC
    net:add(Convolution(256,512,7,7,1,1,0,0))      
    --net:add(BatchNorm(512,1e-5,false))
    net:add(ReLU(true))
    net:add(nn.Dropout())

    -- index 17-19 FC
    net:add(Convolution(512,512,1,1,1,1,0,0))      
    --net:add(BatchNorm(512,1e-5,false))
    net:add(ReLU(true))
    net:add(nn.Dropout())

    -- index 20
    net:add(Convolution(512,nClasses,1,1,1,1,0,0))      
    net:add(nn.View(nClasses))
--    net:add(nn.View(-1,512))
--    net:add(nn.Linear(512,nClasses))      
    net:add(nn.LogSoftMax())

    return net

end

-- Sketch-A-Net + XNOR ----------------------------
function P.sketchanetxnor()

    local net = nn.Sequential()

    -- index 0 is input
    -- index 1
    net:add(Convolution(opts.nChannels,64,15,15,3,3,0,0))      
    net:add(BatchNorm(64,1e-5,false))
    -- index 2
    net:add(ReLU(true))
    -- index 3
    net:add(MaxPooling(3,3,2,2))

    -- index 4-6
    net:add(BinMaxConvolution(64,128,5,5,1,1,0,0))        

    -- index 7-8
    net:add(BinConvolution(128,256,3,3,1,1,1,1))      
    -- index 9-10
    net:add(BinConvolution(256,256,3,3,1,1,1,1)) 
    -- index 11-13
    net:add(BinMaxConvolution(256,256,3,3,1,1,1,1))        

    -- index 14-16 FC
    net:add(Convolution(256,512,7,7,1,1,0,0))      
    net:add(BatchNorm(512,1e-5,false))
    net:add(ReLU(true))
    net:add(nn.Dropout())

    -- index 17-19 FC
    net:add(Convolution(512,512,1,1,1,1,0,0))      
    net:add(BatchNorm(512,1e-5,false))
    net:add(ReLU(true))
    net:add(nn.Dropout())

    -- index 20
    net:add(Convolution(512,nClasses,1,1,1,1,0,0))      

    net:add(nn.View(nClasses))
    net:add(nn.LogSoftMax())

    return net

end

-- Utility functions
function P.updateBinaryGradWeight(convNodes)
    for i =2, #convNodes-1 do
        local n = convNodes[i].weight[1]:nElement()
        local s = convNodes[i].weight:size()
        local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n):expand(s);
        m[convNodes[i].weight:le(-1)]=0;
        m[convNodes[i].weight:ge(1)]=0;
        m:add(1/(n)):mul(1-1/s[2])
        if opts.optimType == 'sgd' then
            m:mul(n);
        end
        convNodes[i].gradWeight:cmul(m)--:cmul(mg)
    end
end

function P.meancenterConvParms(convNodes)
    for i =2, #convNodes-1 do
        local s = convNodes[i].weight:size()
        local negMean = convNodes[i].weight:mean(2):mul(-1):repeatTensor(1,s[2],1,1);  
        convNodes[i].weight:add(negMean)
    end
end


function P.binarizeConvParms(convNodes)
    for i =2, #convNodes-1 do
        local n = convNodes[i].weight[1]:nElement()
        local s = convNodes[i].weight:size()
        local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n);
        convNodes[i].weight:sign():cmul(m:expand(s))
    end
end


function P.clampConvParms(convNodes)
    for i =2, #convNodes-1 do
        convNodes[i].weight:clamp(-1,1)
    end
end

-- Initialization schemes
function P.initialize(scheme)
    if scheme == 'gaussian' 
    or scheme == 'random' then
        error('Initilization scheme not supported')
    elseif scheme == 'xavier' then
        return P.initializeXavier
    end
end

function P.initializeXavier(layer)
    local tn = torch.type(layer)
    if tn == "nn.SpatialConvolution" 
    or tn == "cudnn.SpatialConvolution"
    or tn == "nn.BinarySpatialConvolution" then
        local fanIn  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
        layer.weight:copy(torch.randn(layer.weight:size()) * fanIn)
        layer.bias:fill(0)
        --layer.bias:fill(0.01)
    elseif tn == "nn.Linear" then
        local fanIn =  math.sqrt(2.0 / layer.weight:size(2));
        layer.weight:copy(torch.randn(layer.weight:size()) * fanIn)
        layer.bias:fill(0)
    elseif tn == "nn.SpatialBachNormalization" 
    or  tn == "cudnn.SpatialBachNormalization" then
        layer.weight:fill(1)
        layer.bias:fill(0)
    end
end

return P