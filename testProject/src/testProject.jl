module testProject


mutable struct Node{T}
    hasGrad :: Bool
    value :: T
    grad :: T 
    parents :: Vector{Node}
    back :: Function # funkcja do liczenia gradientu przy przejściu wstecznym

    # hasGrad = true -> Variable
    # hasGard = false -> constant
    function Node(value::T, hasGrad::Bool) where T
        return new{T}(  hasGrad,
                        value,
                        zero(value),
                        Node[],
                        () -> nothing)
    end

    # operator
    function Node(value::T, parents::Vector{Node{T}}, back::Function) where T
        return new{T}(  true,
                        value,
                        zero(value),
                        parents,
                        back)
    end

end


Base.show(io::IO, p::Node) = print(io, "(v=$(p.value), type=", typeof(p),")")

# --------------------------------------------
#           Podstawowe operacje
# --------------------------------------------

import Base: +, -, *, /, sin

function +(x::Node{<:Real}, y::Node{<:Real})
        newNode = Node(x.value + y.value, [x,y], function ()
            x.grad += newNode.grad
            b.grad += newNode.grad
    end)
    return newNode
end



function *(x::Node{<:Real}, y::Node{<:Real})
    newNode = Node(x.value * y.value, [x,y], function ()
        x.grad += y.value * newNode.grad
        y.grad += x.value * newNode.grad
    end)
    return newNode
end

function sin(x::Node)
    newNode = Node(sin(x.value), [x], function ()
        x.grad = cos(x.value) * newNode.grad
    end)
    return newNode
end


function topo_sort(head::Node)
    visited::Set{Node} = Set()
    order::Vector{Node} = Vector()
    visit(head,visited,order)
    return order
end

function visit(head::Node,visited::Set{Node},order::Vector{Node})
    if head.hasGrad

        if head ∉ visited
            push!(visited,head)
            for p in head.parents
                visit(p,visited,order)
            end
            push!(order, head)
        end
    end
end

function backward(head::Node,order::Vector{Node})
    reset(head)
    if head.grad isa Real
        head.grad = one(head.grad) # seed for scalars
    else
        head.grad = ones(size(head.value)) # seed for arrays and matrices
    end
    for n in reverse(order)
        n.back()
    end
end

function reset(head::Node)
    if head.hasGrad
        head.grad = zero(head.grad)
        for p in head.parents
            reset(p)
        end
    end
end


x = Node(5., true)

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
