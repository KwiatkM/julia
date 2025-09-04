include("MLP.jl")

# mutable struct Embedding <: AbstractLayer
#     inputSize::Int64
#     outputSize::Int64
#     build::Function # builds part of the graph associated with current layer, 
#                     # takes input type and current graph head, returns new graph head and pointers for weight nodes

#     function Embedding(inputSize::Int64, outputSize::Int64)
#         layer = new(inputSize,outputSize,activationFunction, 
#         function(type::Type{<:AbstractFloat}, graph::Node)
#             W = Node(randn(type, (layer.outputSize, layer.inputSize)))
#             b = Node(randn(type, (layer.outputSize,1)))
#             newGraph = layer.activationFunction(MLPaddBias((W * graph),b))
#             return newGraph, [W,b]
#         end)
#         return layer
#     end
# end



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
    aaa = true
    pool_size = size(node_grad)[1] ÷ size(pool_grad)[1]
    for batch in 1:size(pool_grad)[3]
        for channel in 1:size(pool_grad)[2]
            for i in 1:size(pool_grad)[1]
                if aaa
                    println(node_grad[(i-1)*pool_size+1:i*pool_size, channel, batch])
                end
                node_grad[(i-1)*pool_size+1:i*pool_size, channel, batch] .*= pool_grad[i,channel,batch]
                if aaa
                    println(node_grad[(i-1)*pool_size+1:i*pool_size, channel, batch])
                end
                aaa = false
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
