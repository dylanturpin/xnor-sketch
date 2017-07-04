require 'paths'
require 'image'
local matio = require 'matio'
local SK = torch.class('sketchDataset')

function SK:__init(matfile)
    self.matfile = matfile or nil
    self:loadFromMatfile()
end

function SK:loadFromMatfile()
    local imdb = matio.load(self.matfile,'imdb')
    self.images = imdb.images:permute(4,3,1,2)
    self.labels = imdb.labels
    self.set    = imdb.set
end

function SK:shuffle()
    local inds = torch.randperm(self.images:size(1):long())
    self.images:index(self.images,1,inds)
    self.labels:index(self.labels,1,inds)
end

function SK:batch(indStart,indEnd)
    local images = self.images[{ indStart, indEnd }]
    local labels = self.labels[{ indStart, indEnd }]
    local sz = images:size()
    if cuda then
        images = images:cuda()
        labels = labels:cuda()
    else
        images = images:float()
        labels = labels:float()
    end
    return images, labels
end

function SK:subset(set)
    local inds
    if set == 'train' then
        inds = self.set:eq(1)
    else if set == 'val' then
        inds = self.set:eq(2)
    else 
        inds = self.set:eq(3)
    end
    -- Create new class instance and copy fields
    local res = SK.new()
    res.matfile = self.matfile
    res.images = self.images[{ inds }]
    res.labels = self.images[{ inds }]
    res.set = self.set[{ inds }]
    return res
end