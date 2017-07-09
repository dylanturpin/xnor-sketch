-- Network architectures
local P = {}; networks = P;
require "nn"

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
    self.output:resizeAs(input):copy(input):sign()
    return self.output
end

function BinActiveZ:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput[input:ge(1)]=0
    self.gradInput[input:le(-1)]=0
    return self.gradInput
end

-- Network architectures-------------------------------------------------
-- Alexnet model ---------------------------------------------------------
function P.alexnet()
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
    net:add(Convolution(4096, opts.nClasses,1,1))
    net:add(nn.View(opts.nClasses))
    net:add(nn.LogSoftMax())

    return net

end

-- Alexnet  + xnor -----------------------------------------
function P.alexnetxnor()
-- Binary convolution
    local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local C= nn.Sequential()
        C:add(BatchNorm(nInputPlane,1e-4,false))
        C:add(nn.BinActiveZ())
        C:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
        return C
    end
    
    local net = nn.Sequential()
    net:add(Convolution(3,96,11,11,4,4,2,2))      
    net:add(BatchNorm(96,1e-5,false))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3,2,2))

    net:add(BinConvolution(96,256,5,5,1,1,2,2))        
    net:add(MaxPooling(3,3,2,2))
    net:add(BinConvolution(256,384,3,3,1,1,1,1))      
    net:add(BinConvolution(384,384,3,3,1,1,1,1)) 
    net:add(BinConvolution(384,256,3,3,1,1,1,1))        
    net:add(MaxPooling(3,3,2,2))
    net:add(BinConvolution(256,4096,6,6))        
    net:add(BinConvolution(4096,4096,1,1))

    net:add(BatchNorm(4096,1e-3,false))
    net:add(ReLU(true))
    net:add(Convolution(4096, opts.nClasses,1,1))

    net:add(nn.View(opts.nClasses))
    net:add(nn.LogSoftMax())

    return net
end

-- Sketch-A-Net ----------------------------------------
function P.sketchnet()
    local function ContConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local C= nn.Sequential()
        C:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
        -- C:add(BatchNorm(nOutputPlane,1e-3))
        C:add(ReLU(true))
        return C
    end

    local net = nn.Sequential()
    -- Layer 1
    net:add(ContConvolution(opts.nChannels,64,15,15,3,3,0,0))      
    net:add(MaxPooling(3,3,2,2))
    -- Layer 2
    net:add(ContConvolution(64,128,5,5,1,1,0,0))      
    net:add(MaxPooling(3,3,2,2))
    -- Layer 3-5
    net:add(ContConvolution(128,256,3,3,1,1,1,1))      
    net:add(ContConvolution(256,256,3,3,1,1,1,1))      
    net:add(ContConvolution(256,256,3,3,1,1,1,1))      
    net:add(MaxPooling(3,3,2,2))
    -- Layer 6 (fully connected)
    net:add(ContConvolution(256,512,7,7,1,1,0,0))      
    net:add(nn.Dropout(0.5))
    -- Layer 7 (fully connected)
    net:add(ContConvolution(512,512,1,1,1,1,0,0))      
    net:add(nn.Dropout(0.5))
    -- Layer 8 (classification)
    net:add(Convolution(512,opts.nClasses,1,1,1,1,0,0))      
    net:add(nn.View(opts.nClasses))
    net:add(nn.LogSoftMax())

    return net

end

-- Sketch-A-Net + XNOR ----------------------------
function P.sketchnetxnor()
    local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        local C= nn.Sequential()
--        C:add(BatchNorm(nInputPlane,1e-4,false))
        C:add(nn.BinActiveZ())
        C:add(Convolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
        return C
    end
    local net = nn.Sequential()
    -- Layer 1 (we keep the normal sequence of layers for this one)
    net:add(Convolution(opts.nChannels,64,15,15,3,3,0,0))      
--    net:add(BatchNorm(96,1e-5,false))
    net:add(ReLU(true))
    net:add(MaxPooling(3,3,2,2))
    -- Layer 2
    net:add(BinConvolution(64,128,5,5,1,1,0,0))      
    net:add(MaxPooling(3,3,2,2))
    -- Layer 3-5
    net:add(BinConvolution(128,256,3,3,1,1,1,1))      
    net:add(BinConvolution(256,256,3,3,1,1,1,1))      
    net:add(BinConvolution(256,256,3,3,1,1,1,1))      
    net:add(MaxPooling(3,3,2,2))
    -- Layer 6 (fully connected)
    net:add(BinConvolution(256,512,7,7,1,1,0,0))      
    net:add(nn.Dropout(0.5))
    -- Layer 7 (fully connected)
    net:add(BinConvolution(512,512,1,1,1,1,0,0))      
    net:add(nn.Dropout(0.5))
    -- Layer 8 (classification)
    net:add(BinConvolution(512,opts.nClasses,1,1,1,1,0,0))      
    net:add(nn.View(opts.nClasses))
    net:add(nn.LogSoftMax())

    return net

end

-- Utility functions -----------------------------------------------------------
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

-- Initialization schemes -----------------------------------------------------
function P.initialize(scheme)
    if scheme == 'gaussian' or scheme == 'random' then
        return P.initializeGaussian
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

function P.initializeGaussian(layer)
    local tn = torch.type(layer)
    if tn == "nn.SpatialConvolution" 
    or tn == "cudnn.SpatialConvolution"
    or tn == "nn.BinarySpatialConvolution" 
    or tn == "nn.Linear" then
        layer.weight:randn(layer.weight:size()):mul(0.01)
        layer.bias:fill(0.01)
    elseif tn == "nn.Linear" then
        layer.weight:randn(layer.weight:size()):mul(0.01)
        layer.bias:fill(0.01)
    elseif tn == "nn.SpatialBachNormalization" 
    or  tn == "cudnn.SpatialBachNormalization" then
        layer.weight:fill(1)
        layer.bias:fill(0)
    end    
end