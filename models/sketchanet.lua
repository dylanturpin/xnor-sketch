function createModel()
   require 'cudnn'

   local function MaxPooling(kW, kH, dW, dH, padW, padH)
      return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   end

   local features = nn.Sequential()
   
   -- index 0 is input
   -- index 1
   features:add(cudnn.SpatialConvolution(opt.nChannels,64,15,15,3,3,0,0))      
   --features:add(nn.SpatialBatchNormalization(64,1e-5,false))
   -- index 2
   features:add(cudnn.ReLU(true))
   -- index 3
   features:add(MaxPooling(3,3,2,2))

   -- index 4-6
   features:add(cudnn.SpatialConvolution(64,128,5,5,1,1,0,0))      
   --features:add(nn.SpatialBatchNormalization(128,1e-5,false))
   features:add(cudnn.ReLU(true))
   features:add(MaxPooling(3,3,2,2))

   -- index 7-8
   features:add(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1))      
   --features:add(nn.SpatialBatchNormalization(256,1e-5,false))
   features:add(cudnn.ReLU(true))

   -- index 9-10
   features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      
   --features:add(nn.SpatialBatchNormalization(256,1e-5,false))
   features:add(cudnn.ReLU(true))

   -- index 11-13
   features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      
   --features:add(nn.SpatialBatchNormalization(256,1e-5,false))
   features:add(cudnn.ReLU(true))
   features:add(MaxPooling(3,3,2,2))

   -- index 14-16 FC
   features:add(cudnn.SpatialConvolution(256,512,7,7,1,1,0,0))      
   --features:add(nn.SpatialBatchNormalization(512,1e-5,false))
   features:add(cudnn.ReLU(true))
   features:add(nn.Dropout())

   -- index 17-19 FC
   features:add(cudnn.SpatialConvolution(512,512,1,1,1,1,0,0))      
   --features:add(nn.SpatialBatchNormalization(512,1e-5,false))
   features:add(cudnn.ReLU(true))
   features:add(nn.Dropout())

   -- index 20
   features:add(cudnn.SpatialConvolution(512,nClasses,1,1,1,1,0,0))      
   
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
   
   local model = features
   return model

end
