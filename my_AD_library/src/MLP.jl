include("my_AD_library.jl")


println("aaaaaaaaaaaaaaaaaaaaaaaaaaa")
abstract type AbstractLayer end

mutable struct FullyConnected <: AbstractLayer
    inputSize::Int64
    outputSize::Int64
    activationFunction::Function
    build::Function # builds part of the graph associated with current layer, 
                    # takes input type and current graph head, returns new graph head and pointers for weight nodes

    function FullyConnected(inputSize::Int64, outputSize::Int64,activationFunction::Function)
        layer = new(inputSize,outputSize,activationFunction, 
        function(type::Type{<:AbstractFloat}, graph::Node)
            W = Node(randn(type, (layer.outputSize, layer.inputSize)))
            b = Node(randn(type, (layer.outputSize,1)))
            newGraph = layer.activationFunction(MLPaddBias((W * graph),b))
            return newGraph, [W,b]
        end)
        return layer
    end
end

mutable struct Loss <:AbstractLayer
    fun::Function
    build::Function # adds final loss function to the graph, 
                            # takes trainYInputNode and current graph head, returns new graph head

    function Loss(fun::Function)
        layer = new(fun, function(trainYInputNode::Node, graph::Node)
            newGraph = layer.fun(trainYInputNode,graph)
            return newGraph
        end)
        return layer
    end

end

mutable struct Model
    head::Node # model loss
    weightNodes::Vector{Node}
    XTrainNode::Node # input for training data
    yTrainNode::Node # input for training expected results
    outputNode::Node # model output
    dataType::Type
end

mutable struct Options
    nEpochs::Int64
    learningRate::Float64
    batchSize::Int64

    function Options()
        return new(100,10e-5,8)
    end
end


function buildModel(type::Type{<:AbstractFloat}, layers::AbstractLayer...)
    trainXInputNode::Node = Node(zeros(type, layers[1].inputSize,1), false)
    trainYInputNode::Node = Node(zeros(type, 1,1),false)
    outputNode::Node = Node(type(0),false) # placeholder value
    weightNodes::Vector{Node} = []
    graph::Node = trainXInputNode
    println("Network structure:")
    for layer in layers
        if layer isa Loss
            outputNode = graph
            graph = layer.build(trainYInputNode,graph)
            println("Loss function: ", layer.fun)
            break
        else
            # graph sizes: activationFunction( W[out x in] * x[in x 1] .+ bias[out x 1])
            newGraph, Ws = layer.build(type, graph)
            graph = newGraph
            append!(weightNodes, Ws)
            println(typeof(layer), ", ", layer.inputSize, " -> ", layer.outputSize, "  ", layer.activationFunction)
        end
    end
    
    return Model(graph,weightNodes,trainXInputNode,trainYInputNode,outputNode,type)
end

function binaryAccuracy(y,ŷ)
    return (y>0.5) == (ŷ>0.5) ? 1. : 0.
end


function train(model::Model, X::AbstractArray{T}, y::AbstractArray{T}, options::Options) where T <: AbstractFloat
    if typeof(X[1]) != model.dataType
        X = model.dataType.(X)
    end
    if typeof(y[1]) != model.dataType
        y = model.dataType.(y)
    end

    model.XTrainNode.value = zeros(model.dataType, size(model.XTrainNode.value,1),options.batchSize)
    model.XTrainNode.grad = zero(model.XTrainNode.value)

    model.yTrainNode.value = zeros(model.dataType, options.batchSize,1)
    model.yTrainNode.grad = zero(model.yTrainNode.value)

    total_loss = 0.
    # total_accuracy = 0.
    oredr = topo_sort(model.head)
    nBatches = div(size(X,2),options.batchSize)
    cols = axes(X,2) # columns of X
    fixGradDim(model,options.batchSize)
    for n in 1:options.nEpochs
        total_loss = 0.
        # total_accuracy = 0.
        t = @elapsed begin
            for n in 1:nBatches
                model.XTrainNode.value = X[:,(n-1)*options.batchSize+1:n*options.batchSize]
                model.yTrainNode.value = y[(n-1)*options.batchSize+1:n*options.batchSize,:]'
                forward(model.head)
                # println("output value: ",size(model.outputNode.value))
                # println("loss value: ",size(model.head.value))
                # println("output gradient: ",size(model.outputNode.grad))
                # println("loss gradient: ",size(model.head.grad))

                backward(model.head,oredr)
                total_loss += sum(model.head.value)
                
                # total_accuracy += model.accuracy(y[col], model.head.value)
                for w in model.weightNodes

                    w.value = w.value .- options.learningRate .* w.grad
                end
            end
        end
        # t = @elapsed begin
        #     for col in axes(X,2)
        #         model.XTrainNode.value = X[:,col]
        #         model.yTrainNode.value = y[col]
        #         forward(model.head)
        #         backward(model.head,oredr)
        #         total_loss += model.head.value
                
        #         # total_accuracy += model.accuracy(y[col], model.head.value)
        #         for w in model.weightNodes

        #             w.value -= options.learningRate .* w.grad
        #         end
        #     end
        # end
        # print("Epoch: ", n, "\t time: ", t, "s\t loss: ", total_loss/size(X, 2), "\t acc: ", total_accuracy/size(X, 2), "\n")
        print("Epoch: ", n, "\t time: ", t, "s\t loss: ", total_loss/size(X, 2), "\n")
    end
    
end

function inference(model::Model, X::Vector)
    model.outputNode.value = zeros(model.dataType, 1, 1)
    X = reshape(X,length(X),1)
    model.XTrainNode.value = X
    forward(model.head)
    return model.outputNode.value
end

function fixGradDim(model::Model, batchSize::Int64)
    sampleData = zeros(model.dataType, size(model.XTrainNode.value,1),batchSize)
    model.XTrainNode.value = sampleData
    forward(model.head)
    checkGrad(model.head)
end
function checkGrad(node::Node)
    if size(node.value) != size(node.grad)
        node.grad = zero(node.value)
    end
    for n in node.parents
        checkGrad(n)
    end
end

