include("my_AD_library.jl")



abstract type AbstractLayer end

mutable struct FullyConnected <: AbstractLayer
    inputSize::Int64
    outputSize::Int64
    activationFunction::Function
end

# assumes the input data is in Float64 type
function buildModel(layers::AbstractLayer...)
    dataInputNode::Node = Node(zeros(layers[1].inputSize), false)
    wasGraphInit = false
    graph::Node = Node(0.)
    for layer in layers
        # graph = activationFunction( W[out x in] * x[in x 1] + bias[out x 1])
        if !wasGraphInit
            graph = layer.activationFunction(Node(randn(layer.outputSize, layer.inputSize))*
                    dataInputNode+
                    Node(randn(layer.outputSize)))
            wasGraphInit = true
        else
            # TODO: check if output size form previous node matches the input size of the current one
            graph = layer.activationFunction(Node(randn(layer.outputSize, layer.inputSize))*
                    graph +
                    Node(randn(layer.outputSize)))           
        end

    end
    return graph
end



buildModel(
    FullyConnected(10,12,tanh),
    FullyConnected(12,5,ReLU))


