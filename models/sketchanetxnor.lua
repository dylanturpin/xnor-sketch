function createModel()
   require 'cudnn'
   local function activation()
      local C= nn.Sequential()
      C:add(nn.BinActiveZ())
      return C
   end

   local function MaxPooling(kW, kH, dW, dH, padW, padH)
      return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   end

   local function BinConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
      local C= nn.Sequential()
      C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
      C:add(activation())
      C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
      return C
   end

   local function BinMaxConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,mW,mH)
      local C= nn.Sequential()
       C:add(nn.SpatialBatchNormalization(nInputPlane,1e-4,false))
      C:add(activation())
      C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
      C:add(MaxPooling(3,3,2,2))
      return C
   end
   
   local features = nn.Sequential()
   
   -- index 0 is input
   -- index 1
   features:add(cudnn.SpatialConvolution(opt.nChannels,64,15,15,3,3,0,0))      
   features:add(nn.SpatialBatchNormalization(64,1e-5,false))
   -- index 2
   features:add(cudnn.ReLU(true))
   -- index 3
   features:add(MaxPooling(3,3,2,2))

   -- index 4-6
   features:add(BinMaxConvolution(64,128,5,5,1,1,0,0))        

   -- index 7-8
   features:add(BinConvolution(128,256,3,3,1,1,1,1))      
   -- index 9-10
   features:add(BinConvolution(256,256,3,3,1,1,1,1)) 
   -- index 11-13
   features:add(BinMaxConvolution(256,256,3,3,1,1,1,1))        

   -- index 14-16 FC
   features:add(cudnn.SpatialConvolution(256,512,7,7,1,1,0,0))      
   features:add(nn.SpatialBatchNormalization(512,1e-5,false))
   features:add(cudnn.ReLU(true))
   features:add(nn.Dropout())

   -- index 17-19 FC
   features:add(cudnn.SpatialConvolution(512,512,1,1,1,1,0,0))      
   features:add(nn.SpatialBatchNormalization(512,1e-5,false))
   features:add(cudnn.ReLU(true))
   features:add(nn.Dropout())

   -- index 20
   features:add(cudnn.SpatialConvolution(512,nClasses,1,1,1,1,0,0))      
   
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
   
   local model = features
   return model

end
