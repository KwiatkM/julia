

mutable struct Node{T}
    hasGrad :: Bool
    value :: T
    grad :: T 
    parents :: Vector{Node}
    back :: Function

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
    function Node(value::T, parents::Vector{<:Node}, back::Function) where T
        return new{T}(  true,
                        value,
                        zero(value),
                        parents,
                        back)
    end

end


Base.show(io::IO, p::Node) = print(io, "(v=$(p.value), grad=$(p.grad))")

# --------------------------------------------
# --------------------------------------------
#           Basic operations
# --------------------------------------------
# --------------------------------------------


import Base: +, -, *, /, sin, sum

# --------------------------------------------
#           Addition
# --------------------------------------------

function +(x::Node{<:Real}, y::Node{<:Real})
        newNode = Node(x.value + y.value, [x,y], function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad += newNode.grad end
    end)
    return newNode
end

function +(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
        newNode = Node(x.value + y.value, [x,y], function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad += newNode.grad end
    end)
    return newNode
end

# --------------------------------------------------
#           Multiplication
# --------------------------------------------------

function *(x::Node{<:Real}, y::Node{<:Real})
    newNode = Node(x.value * y.value, [x,y], function ()
        if x.hasGard x.grad += y.value * newNode.grad end
        if y.hasGard y.grad += x.value * newNode.grad end
    end)
    return newNode
end

function *(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    newNode = Node(x.value * y.value, [x,y], function ()
        if x.hasGrad x.grad += newNode.grad * y.value' end
        if y.hasGrad y.grad += x.value' * newNode.grad end
    end)
    return newNode
end

function mult_ew(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    newNode = Node(x.value .* y.value, [x,y], function ()
        if x.hasGrad x.grad += newNode.grad .* y.value end
        if x.hasGrad y.grad += newNode.grad .* x.value end
    end)
    return newNode
end

# --------------------------------------------------
#           trigonometry
# --------------------------------------------------

function sin(x::Node{<:Real})
    newNode = Node(sin(x.value), [x], function ()
        if x.hasGrad x.grad = cos(x.value) * newNode.grad end
    end)
    return newNode
end

function sin(x::Node{<:AbstractArray})
    newNode = Node(sin(x.value), [x], function ()
        if x.hasGrad x.grad = cos(x.value) .* newNode.grad end
    end)
    return newNode
end

# --------------------------------------------------
#           other
# --------------------------------------------------

function sum(x::Node{<:AbstractArray})
    newNode = Node(sum(x.value), [x], function ()
        if x.hasGrad x.grad .+= newNode.grad end
    end)
    return newNode
end

# --------------------------------------------------





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

function backward(head::Node{T},order::Vector{Node}, seed::T) where {T}
    reset(head)
    head.grad = seed
    for n in reverse(order)
        n.back()
    end
end

function backward(head::Node,order::Vector{Node})

    if head.grad isa Real
        backward(head,order,one(head.grad)) # default seed for scalars
    else
        backward(head,order,ones(size(head.value)))# default seed for arrays and matrices
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



