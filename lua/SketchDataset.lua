require 'paths'
require 'image'
local matio = require 'matio'
local SketchDataset = torch.class('SketchDataset')

function SketchDataset:__init(matfile)
    if matfile then
        if type(matfile) == 'string' then
            if paths.filep(matfile) then
                self.matfile = matfile
                self:loadFromMatfile(matfile)
            else
                error('File was not found')
            end
        else
            error('File path is not a string')
        end
    else
        self.images = nil
        self.labels = nil
        self.set = nil
    end
end

function SketchDataset:loadFromMatfile(matfile)
    local imdb = matio.load(matfile,'imdb')
    self.images = imdb.images.data:permute(4,3,1,2)
    self.labels = imdb.images.labels:view(-1) -- flatten vectors
    self.set    = imdb.images.set:view(-1)
    -- Sanity checks
    assert(self.images:size(2) == 1) 
    assert(self.images:size(1) == self.labels:nElement())
end

-- Create a new class instance for one of the train,val,test subsets.
function SketchDataset:split()    
    self.train  = {}
    self.val    = {}
    self.test   = {}
    -- The dataset contains 20K images: 13500 for training and 6500 for testing.
    -- Set label '1' is for training, and '3' for testing.
    self.train.images = self.images:index(1,self.set:eq(1):nonzero():view(-1))
    self.train.labels = self.labels[self.set:eq(1)]
    self.test.images  = self.images:index(1,self.set:eq(3):nonzero():view(-1))
    self.test.labels  = self.labels[self.set:eq(3)]
    -- Save space in memory
    self.images = nil
    self.labels = nil
    -- We randomly shuffle training data and keep the last 1500 for validation
    local shuffle = torch.randperm(self.train.images:size(1)):long()
    self.val.images = self.train.images:index(1, shuffle[{ {12001,13500} }])
    self.val.labels = self.train.labels:index(1, shuffle[{ {12001,13500} }])
    self.train.images = self.train.images:index(1, shuffle[{ {1,12000} }])
    self.train.labels = self.train.labels:index(1, shuffle[{ {1,12000} }])
    collectgarbage()
end

-- Data augmentation (takes place only on the training set)
function SketchDataset:augment()
   -- Crops
   
   -- Rotations
   
   -- Horizontal flip
end
