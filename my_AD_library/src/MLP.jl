include("my_AD_library.jl")



abstract type AbstractLayer end

mutable struct FullyConnected <: AbstractLayer
    inputSize::Int64
    outputSize::Int64
    activationFunction::Function
end

mutable struct Loss <:AbstractLayer
    fun::Function
end

mutable struct Model
    head::Node
    weightNodes::Vector{Node}
    XTrainNode::Node
    yTrainNode::Node
    # accuracy::Function
end

mutable struct Options
    nEpochs::Int64
    learningRate::Float64

end

# TODO: check if output size form previous node matches the input size of the current one
# assumes the input data is in Float32 type
# TODO: add different data types
function buildModel(layers::AbstractLayer...)
    trainXInputNode::Node = Node(zeros(Float32, layers[1].inputSize), false)
    trainYInputNode::Node = Node(.0f0,false)
    weightNodes::Vector{Node} = []
    graph::Node = trainXInputNode
    for layer in layers
        if layer isa Loss
            graph = layer.fun(trainYInputNode,graph)
        else
            # graph sizes: activationFunction( W[out x in] * x[in x 1] + bias[out x 1])
            W = Node(randn(Float32, (layer.outputSize, layer.inputSize)))
            b = Node(randn(Float32, layer.outputSize))
            graph = layer.activationFunction((W * graph) + b)
            push!(weightNodes, W, b)
        end
    end
    
    return Model(graph,weightNodes,trainXInputNode,trainYInputNode)
end

function binaryAccuracy(y,ŷ)
    return (y>0.5) == (ŷ>0.5) ? 1. : 0.
end

# TODO: add batch training. Right now it's stochastic gradient descent
function train(model::Model, X::AbstractArray{Float32}, y::AbstractArray{Float32}, options::Options)
    total_loss = 0.
    # total_accuracy = 0.
    oredr = topo_sort(model.head)
    for n in 1:options.nEpochs
        total_loss = 0.
        # total_accuracy = 0.
        t = @elapsed begin
        for col in axes(X,2)
            model.XTrainNode.value = X[:,col]
            model.yTrainNode.value = y[col]
            forward(model.head)
            backward(model.head,oredr)
            total_loss += model.head.value
            
            # total_accuracy += model.accuracy(y[col], model.head.value)
            for w in model.weightNodes

                w.value -= options.learningRate .* w.grad
            end
        end
        end
        # print("Epoch: ", n, "\t time: ", t, "s\t loss: ", total_loss/size(X, 2), "\t acc: ", total_accuracy/size(X, 2), "\n")
        print("Epoch: ", n, "\t time: ", t, "s\t loss: ", total_loss/size(X, 2), "\n")
    end
    
end

function inference(model::Model, X::Vector)
    model.XTrainNode.value = X
    forward(model.head)
    return model.head.parents[2].value
end

