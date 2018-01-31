--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--
--  modified by Yue Wu(email: yuewu@ece.neu.edu) 2016
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'cutorch'
cutorch.setDevice(1)
local t = require './transforms'

-- model 6 100k
local imagenetLabel6 = require './imagenet'
local model6 = torch.load(arg[1]):cuda()

-- local softMaxLayer = cudnn.SoftMax()
local softMaxLayer = cudnn.SoftMax():cuda()
model6:add(softMaxLayer)
model6:evaluate()




-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.TwoCrop(224),
}


function file_exists(file)
	local f = io.open(file, "rb")
	if f then f:close() end
	return f ~= nil
end

function lines_from(file)
	if not file_exists(file) then return {} end
	lines = {}
	for line in io.lines(file) do 
		lines[#lines + 1] = line
	end
	return lines
end


-- take one line as test
local file = arg[2]
local lines = lines_from(file)



local N = 1
-- local N1 = 1


-- for sort
function spairs(t, order)
	local keys = {}
	for k in pairs(t) do keys[#keys+1] = k end

	if order then
		table.sort(keys, function(a,b) return order(t, a, b) end)
	else
		table.sort(keys)
	
	end

	local i = 0
	return function()
		i = i + 1
		if keys[i] then
			return keys[i], t[keys[i]]
		end
	end
end

function compare(t,a,b) 
	return t[b] < t[a] 
end



function setContains(set, key)
	return set[key] ~= nil
end


local threshold = 0.95
-- for i=2,#arg do
for k,v in pairs(lines) do
   -- load the image as a RGB float tensor with values 0..1
   -- local img = image.load(arg[i], 3, 'float')
   local begintime = os.clock()

   -- get batch 
   local img = image.load(v, 3, 'float')
   img = transform(img)
   local imageSize = img:size():totable()
   table.remove(imageSize, 1)
   local batch = img:view(2, table.unpack(imageSize))

   -- Get the output of the softmax for model 1 to 5
   local allres = {}
   local allres_6 = {}

   local output6 = model6:forward(batch:cuda()):squeeze()
   output6 = output6:view(output6:size(1)/2, 2, output6:size(2)):sum(2):squeeze():div(2):float()
   local outfile = v .. '.mat'
   print (outfile)
   -- print (output6)
   local matio = require 'matio'
   matio.save(outfile, output6)
   
   local probs6, indexes6 = output6:topk(N, true, true)
   for n=1,N do
	   allres_6[ imagenetLabel6[indexes6[n]]] = probs6[n]
   end

  local topprob_6 = allres_6[imagenetLabel6[indexes6[1]]]

     -- local endtime = os.clock()
   -- print(string.format("forward time: %.2f\n", endtime- begintime))
end



