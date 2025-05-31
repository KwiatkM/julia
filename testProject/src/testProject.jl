module testProject

abstract type Node end

# ConstantNode i VariableNode mogą być tylko liścmi grafu

struct ConstantNode{F} <: Node
    value::F
end

mutable struct VariableNode{F} <: Node
    value :: F
    grad :: F
end

mutable struct OperatorNode{F} <: Node
    value :: F
    grad :: F 
    parents :: Vector{Node}
    vjp :: Function
end


Base.show(io::IO, p::Node) = print(io, "(v=$(p.value), type=", typeof(p),")")

# --------------------------------------------
#           Podstawowe operacje
# --------------------------------------------

import Base: +, -, *, /, sin

function +(x::Node, y::Node)
        newNode = OperatorNode{Float64}(x.value + y.value,0., [x,y], function ()
        if typeof(x) != ConstantNode x.grad += newNode.grad end
        if typeof(y) != ConstantNode y.grad += newNode.grad end
    end)
    println(newNode)
    return newNode
end

function *(x::Node, y::Node)
    newNode = OperatorNode{Float64}(x.value * y.value,0., [x,y], function ()
        if typeof(x) != ConstantNode x.grad += y.value * newNode.grad end
        if typeof(y) != ConstantNode y.grad += x.value * newNode.grad end
    end)
    println(newNode)
    return newNode
end

function sin(x::Node)
    newNode = OperatorNode{Float64}(sin(x.value),0., [x], function ()
        if typeof(x) != ConstantNode x.grad = cos(x.value) * newNode.grad end
    end)
    println(newNode)
    return newNode
end


function topo_sort(head::Node)
    
    visited::Set{Node} = Set()
    order::Vector{Node} = Vector()
    visit(head,visited,order)
    return order
end

function visit(head::Node,visited::Set{Node},order::Vector{Node})
    if head isa OperatorNode 

        if head ∉ visited
            push!(visited,head)
            for p in head.parents
                visit(p,visited,order)
            end
            push!(order, head)
        end
    end
end

function backward(head::OperatorNode{Float64},order::Vector{Node})
    reset(head)
    head.grad = 1.
    for n in reverse(order)
        n.vjp()
    end
end

function reset(head::OperatorNode{Float64})
    head.grad = 0.
    for p in head.parents
        reset(p)
    end
end

function reset(head::VariableNode{Float64})
    head.grad = 0.
end

function reset(head::ConstantNode{Float64})
    nothing
end



x = VariableNode{Float64}(5.,0.)

f = sin(x*x)

order = topo_sort(f)

println("\nResult: ",f)
println("Order: ", order)

backward(f,order)
println("x grad = ", x.grad)

backward(f,order)
println("x grad = ", x.grad)

# println("\nend of test")
end # module testProject
