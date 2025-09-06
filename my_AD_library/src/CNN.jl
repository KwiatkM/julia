include("MLP.jl")

abstract type AbstractDataTransformLayer <: AbstractLayer end

mutable struct Embedding <: AbstractDataTransformLayer
    inputSize::Int64
    outputSize::Int64
    build::Function # builds part of the graph associated with current layer, 
                    # takes input type and current graph head, returns new graph head and pointers for weight nodes
    printInfo::Function

    function Embedding(inputSize::Int64, embeddings::AbstractMatrix)
        layer = new(inputSize, size(embeddings)[1], 
        function(type::Type{<:AbstractFloat}, graph::Node)
            newGraph = dataMap(graph, embeddings)
            return newGraph, []
        end,
        function()
            println("Embedding, ", layer.inputSize," -> ",layer.outputSize, "x", layer.inputSize)
        end)
        return layer
    end
end

mutable struct PermuteDims <: AbstractDataTransformLayer
    permutations::Tuple
    build::Function # builds part of the graph associated with current layer, 
                    # takes input type and current graph head, returns new graph head and pointers for weight nodes
    printInfo::Function
       
    function PermuteDims(permutations::Tuple)
        layer = new( permutations,
        function(type::Type{<:AbstractFloat}, graph::Node)
            newGraph = myPermutedims(graph,permutations)
            return newGraph, []
        end,
        function()
            println("PermuteDims, ", layer.permutations)
        end)
        return layer
    end
end

mutable struct Convolution_1d <: AbstractLayer
    inputChannels::Int64
    outputChannels::Int64
    kernelSize::Int64
    activationFunction::Function
    build::Function # builds part of the graph associated with current layer, 
                    # takes input type and current graph head, returns new graph head and pointers for weight nodes
    printInfo::Function

    function Convolution_1d(inputChannels::Int64, outputChannels::Int64,kernelSize::Int64, activationFunction::Function)
        layer = new(inputChannels,outputChannels,kernelSize, activationFunction, 
        function(type::Type{<:AbstractFloat}, graph::Node)

            kernelNode = Node(randn(type, layer.outputChannels, layer.inputChannels * layer.kernelSize))
            bias = Node(randn(type, layer.outputChannels,1))
            step1 = im2col(graph,kernelSize)
            step2 = kernelNode * step1
            step3 = MLPaddBias(step2,bias)
            step4 = layer.activationFunction(step3)
            newGraph = convResultReshape(step4, size(graph.value)[1] - layer.kernelSize + 1) #second value = data length after convolution

            return newGraph, [kernelNode, bias]
        end,
        function()
            println("Convolution_1d, k=", layer.kernelSize, ", ", layer.inputChannels, " -> ", layer.outputChannels, ", ", layer.activationFunction)
        end)
        return layer
    end
end

mutable struct MaxPool_1d <: AbstractLayer
    kernelSize::Int64
    build::Function # builds part of the graph associated with current layer, 
                    # takes input type and current graph head, returns new graph head and pointers for weight nodes
    printInfo::Function

    function MaxPool_1d(kernelSize::Int64)
        layer = new(kernelSize, 
        function(type::Type{<:AbstractFloat}, graph::Node)
            newGraph = max_pool_1d(graph,kernelSize)

            return newGraph, []
        end,
        function()
            println("MaxPool_1d, ", kernelSize)
        end)
        return layer
    end
end


mutable struct myFlatten <: AbstractLayer
    build::Function # builds part of the graph associated with current layer, 
                    # takes input type and current graph head, returns new graph head and pointers for weight nodes
    printInfo::Function

    function myFlatten()
        layer = new( 
        function(type::Type{<:AbstractFloat}, graph::Node)
            newGraph = flatten(graph)

            return newGraph, []
        end,
        function()
            println("Flatten")
        end)
        return layer
    end
end



function im2col(x::Node{<:AbstractArray{T}},kernel_size::Integer, stride::Integer=1) where T <: AbstractFloat
    newNode = Node(im2col(x.value,kernel_size,stride), [x],function ()
            newNode.value = im2col(newNode.parents[1].value, kernel_size, stride)
        end, 
        function ()
            nothing
    end, "im2col")
    return newNode
end

function convResultReshape(x::Node{<:AbstractArray{T}}, dataSize::Integer) where T <: AbstractFloat
    newNode = Node(im2col_fixResultDim(x.value,dataSize), [x],function ()
            newNode.value = im2col_fixResultDim(newNode.parents[1].value, dataSize)
        end, 
        function ()
            if x.hasGrad x.grad = im2col_reverseResultDim(newNode.grad) end
    end, "convResultReshape")
    return newNode
end

function max_pool_1d(x::Node{<:AbstractArray{T}}, windowSize::Integer) where T <: AbstractFloat
    newNode = Node(max_pool_1d!(x.value, x.grad, windowSize), [x],function ()
            newNode.value = max_pool_1d!(newNode.parents[1].value, newNode.parents[1].grad,windowSize)
        end, 
        function ()
            if x.hasGrad backprop_max_pool_1d!(newNode.grad, x.grad) end
    end, "max_pool_1d")
    return newNode
end

function flatten(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(reshape(x.value, size(x.value)[1]*size(x.value)[2], size(x.value)[3]), [x],function ()
            newNode.value = reshape(x.value, size(x.value)[1]*size(x.value)[2], size(x.value)[3])
        end, 
        function ()
            if x.hasGrad x.grad = reshape(newNode.grad, size(x.grad)) end
    end, "flatten")
    return newNode    
end

# image size: 5
# chanels (input): 4
# batch size: 2


# image dim x chanels x batch size (e.g. 130x50x3)
function im2col(data::Array{T, 3}, kernel_size::Integer, stride::Integer=1) where T
    data_in_size, chanels, batch_size = size(data)
    data_out_size = (data_in_size - kernel_size) ÷ stride + 1

    cols = Array{T}(undef, chanels * kernel_size, batch_size * data_out_size)

    row = 1
    for b in 1:batch_size
        for i in 1:stride:(data_in_size - kernel_size + 1)
            patch = vec(data[i:i + kernel_size - 1, :, b])
            cols[:, row] .= patch
            row += 1
        end
    end
    
    return cols#, data_out_size
end

# for forward pass
function im2col_fixResultDim(result::Array{T, 2}, data_size::Integer) where T
    batch_size = size(result)[2] ÷ data_size
    return permutedims(reshape(result, size(result)[1], data_size, batch_size), (2,1,3))
end

# for gradient calculations
function im2col_reverseResultDim(data::Array{T, 3}) where T
    L,Ch,B = size(data)

    return reshape(permutedims(data,(2,1,3)),Ch,L*B)

end

# TODO: zapisać która wartość została wybrana jako największa (do liczenia gradientu) 
function max_pool_1d(data::Array{T, 3}, kernel_size) where T
    batch_size = size(data)[3]
    out_data_size = size(data)[1] ÷ kernel_size
    output = Array{T}(undef, out_data_size, size(data)[2], batch_size)
    for batch in 1:batch_size
        for channel in 1:size(data)[2]
            for i in 1:out_data_size
                output[i,channel,batch] = maximum(data[(i-1)*kernel_size+1:i*kernel_size, channel,batch])
            end
        end
    end
    return output
end

function max_pool_1d!(data::Array{T, 3}, mask::Array{T,3}, kernel_size) where T
    if size(data) != size(mask)
        mask = zeros(size(data))
    end
    batch_size = size(data)[3]
    out_data_size = size(data)[1] ÷ kernel_size
    output = Array{T}(undef, out_data_size, size(data)[2], batch_size)
    for batch in 1:batch_size
        for channel in 1:size(data)[2]
            for i in 1:out_data_size
                max_val, max_index = findmax(data[(i-1)*kernel_size+1:i*kernel_size, channel,batch])
                output[i,channel,batch] = max_val
                mask[(i-1)*kernel_size+1:i*kernel_size, channel,batch] .= 0
                mask[(i-1)*kernel_size + max_index, channel,batch] = 1
            end
        end
    end
    return output
end



function backprop_max_pool_1d!(pool_grad::Array{T, 3}, node_grad::Array{T, 3}) where T
    pool_size = size(node_grad)[1] ÷ size(pool_grad)[1]
    for batch in 1:size(pool_grad)[3]
        for channel in 1:size(pool_grad)[2]
            for i in 1:size(pool_grad)[1]
                node_grad[(i-1)*pool_size+1:i*pool_size, channel, batch] .*= pool_grad[i,channel,batch]
            end
        end
    end
end



# # A = zeros(5,2,2)
# A[:,:,1] = [1 2; 3 4; 5 6; 7 8; 9 10]

# R = im2col(A,3)
# k = randn(3,6)

# t = k * R[1]
# tt = reshape(t, 4,3,2)
# ttt = permutedims(tt, (2,1,3))
