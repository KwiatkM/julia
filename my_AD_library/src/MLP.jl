include("my_AD_library.jl")



abstract type AbstractLayer end

mutable struct FullyConnected <: AbstractLayer
    inputSize::Int64
    outputSize::Int64
    activationFunction::Function
end



mutable struct Model
    head::Node
    weightNodes::Vector{Node}
    dataNode::Node
    accuracy::Function
    # X::AbstractArray{Float64}
    # Y::AbstractArray{Float64}
end


# assumes the input data is in Float32 type
# input as column vector
function buildModel(layers::AbstractLayer...)
    dataInputNode::Node = Node(zeros(Float32, layers[1].inputSize), false)
    weightNodes::Vector{Node} = []
    wasGraphInit = false
    graph::Node = dataInputNode
    for layer in layers
        # graph = activationFunction( W[out x in] * x[in x 1] + bias[out x 1])
        if !wasGraphInit
            W = Node(randn(Float32, (layer.outputSize, layer.inputSize)))
            b = Node(randn(Float32, layer.outputSize))
            graph = layer.activationFunction(W * dataInputNode + b)
            wasGraphInit = true
            push!(weightNodes, W, b)
        else
            # TODO: check if output size form previous node matches the input size of the current one
            W = Node(randn(Float32, (layer.outputSize, layer.inputSize)))
            b = Node(randn(Float32, layer.outputSize))            
            graph = layer.activationFunction(W * graph + b)
            push!(weightNodes, W, b)           
        end
    end
    return graph, weightNodes, dataInputNode
end

function binaryAccuracy(y,ŷ)
    return (y>0.5) == (ŷ>0.5) ? 1. : 0.
end


function train(model::Model, X::AbstractArray{Float32}, y::AbstractArray{Float32}, epochs::Int64)
    α = 0.1f0
    total_loss = 0.
    total_accuracy = 0.
    y_true = Node(0.0f0,false)
    f = myBinaryCrossEntropy(y_true,model.head)
    oredr = topo_sort(f)
    for n in 1:epochs
        total_loss = 0.
        total_accuracy = 0.
        t = @elapsed begin
        for col in axes(X,2)
            model.dataNode.value = X[:,col]
            y_true.value = y[col]
            forward(f)
            backward(f,oredr)
            total_loss += f.value
            
            total_accuracy += model.accuracy(y[col], f.value)
            for w in model.weightNodes
                # println("value:", size(w.value))
                # println("grad:", size(w.grad))
                w.value -= α .* w.grad
            end
        end
        end
        print("Epoch: ", n, "\t time: ", t, "s\t loss: ", total_loss/size(X, 2), "\t acc: ", total_accuracy/size(X, 2), "\n")
    end
    
end

