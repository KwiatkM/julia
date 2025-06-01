

mutable struct Node{T}
    hasGrad :: Bool
    value :: T
    grad :: T 
    parents :: Vector{Node}
    forward::Function
    back :: Function

    # hasGrad = true -> Variable
    # hasGard = false -> constant
    function Node(value::T, hasGrad::Bool=true) where T
        return new{T}(  hasGrad,
                        value,
                        zero(value),
                        Node[],
                        () -> nothing,
                        () -> nothing)
    end

    # operator
    function Node(value::T, parents::Vector{<:Node},forward::Function, back::Function) where T
        return new{T}(  true,
                        value,
                        zero(value),
                        parents,
                        forward,
                        back)
    end

end


Base.show(io::IO, p::Node) = print(io, "(v=$(p.value), grad=$(p.grad))")

# --------------------------------------------
# --------------------------------------------
#           Basic operations
# --------------------------------------------
# --------------------------------------------


import Base: +, -, *, /, sin, cos, tan, tanh, sum

# --------------------------------------------
#           Addition
# --------------------------------------------

function +(x::Node{<:Real}, y::Node{<:Real})
        newNode = Node(x.value + y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] + newNode.parents.value[2] 
        end,
            function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad += newNode.grad end
        end)
    return newNode
end

function +(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
        newNode = Node(x.value + y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] + newNode.parents.value[2] 
        end,
            function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad += newNode.grad end
    end)
    return newNode
end

# --------------------------------------------
#           Subtraction
# --------------------------------------------

function -(x::Node{<:Real}, y::Node{<:Real})
        newNode = Node(x.value - y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] - newNode.parents.value[2] 
        end, 
        function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad -= newNode.grad end
    end)
    return newNode
end

function -(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
        newNode = Node(x.value - y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] - newNode.parents.value[2] 
        end, 
        function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad -= newNode.grad end
    end)
    return newNode
end

function -(x::Node{<:Real})
        newNode = Node(-x.value, [x],function ()
            newNode.value = -newNode.parents.value[1]
        end,  function ()
            if x.hasGrad x.grad -= newNode.grad end

    end)
    return newNode
end

function -(x::Node{<:AbstractArray})
        newNode = Node(-x.value, [x],function ()
            newNode.value = -newNode.parents.value[1]
        end, function ()
            if x.hasGrad x.grad -= newNode.grad end

    end)
    return newNode
end

# --------------------------------------------------
#           Multiplication
# --------------------------------------------------

function *(x::Node{<:Real}, y::Node{<:Real})
    newNode = Node(x.value * y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] * newNode.parents.value[2] 
        end, 
        function ()
        if x.hasGrad x.grad += y.value * newNode.grad end
        if y.hasGrad y.grad += x.value * newNode.grad end
    end)
    return newNode
end

function *(x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    newNode = Node(x.value * y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] * newNode.parents.value[2] 
        end, 
        function ()
        if x.hasGrad x.grad += newNode.grad * y.value' end
        if y.hasGrad y.grad += x.value' * newNode.grad end
    end)
    return newNode
end


function Broadcast.broadcasted(::typeof(*), x::Node{<:AbstractArray}, y::Node{<:AbstractArray})
    newNode = Node(x.value .* y.value, [x,y],function ()
            newNode.value = newNode.parents.value[1] .* newNode.parents.value[2] 
        end, 
        function ()
        if x.hasGrad x.grad += newNode.grad .* y.value end
        if x.hasGrad y.grad += newNode.grad .* x.value end
    end)
    return newNode
end

# --------------------------------------------------
#           trigonometry
# --------------------------------------------------

function sin(x::Node{<:Real})
    newNode = Node(sin(x.value), [x],function ()
            newNode.value = sin(newNode.parents.value[1])
        end,
        function ()
        if x.hasGrad x.grad += cos(x.value) * newNode.grad end
    end)
    return newNode
end

function sin(x::Node{<:AbstractArray})
    newNode = Node(sin.(x.value), [x],function ()
            newNode.value = sin.(newNode.parents.value[1])
        end,
        function ()
        if x.hasGrad x.grad .+= cos.(x.value) .* newNode.grad end
    end)
    return newNode
end

function cos(x::Node{<:Real})
    newNode = Node(cos(x.value), [x],function ()
            newNode.value = cos(newNode.parents.value[1])
        end,
        function ()
        if x.hasGrad x.grad -= sin(x.value) * newNode.grad end
    end)
    return newNode
end

function cos(x::Node{<:AbstractArray})
    newNode = Node(cos.(x.value), [x],function ()
            newNode.value = cos.(newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad .-= sin.(x.value) .* newNode.grad end
    end)
    return newNode
end

function tan(x::Node{<:Real})
    newNode = Node(tan(x.value), [x],function ()
            newNode.value = tan(newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad += 1 / (cos(x.value)^2) * newNode.grad end
    end)
    return newNode
end

function tan(x::Node{<:AbstractArray})
    newNode = Node(tan.(x.value), [x],function ()
            newNode.value = tan.(newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad .+= 1 ./ (cos.(x.value).^2) .* newNode.grad end
    end)
    return newNode
end


# --------------------------------------------------
#           activation functions
# --------------------------------------------------

function tanh(x::Node{<:Real})
    newNode = Node(tanh(x.value), [x],function ()
            newNode.value = tanh(newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad += (1 - newNode.value^2) * newNode.grad end
    end)

    return newNode
end

function tanh(x::Node{<:AbstractArray})
    newNode = Node(tanh.(x.value), [x],function ()
            newNode.value = tanh.(newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad .+= (1 .- newNode.value.^2) .* newNode.grad end
    end)
    return newNode
end

function sigm(x::Node{<:Real})
    newNode = Node(1/(1+ℯ^x.value), [x],function ()
            newNode.value = 1/(1+ℯ^newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad += newNode.value * (1-newNode.value) * newNode.grad end
    end)
    return newNode
end

function sigm(x::Node{<:AbstractArray})
    newNode = Node(1 ./ (1 .+ ℯ.^x.value), [x],function ()
            newNode.value = 1 ./(1 .+ℯ.^newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad .+= newNode.value .* (1 .-newNode.value) .* newNode.grad end
    end)
    return newNode
end

function ReLU(x::Node{<:Real})
    newNode = Node(max(0,x.value), [x],function ()
            newNode.value = max(0,newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad += x.value > zero(x.value) ? newNode.grad : zero(x.value) end
    end)
    return newNode
end

function ReLU(x::Node{<:AbstractArray})
    newNode = Node(max.(0,x.value), [x],function ()
            newNode.value = max.(0,newNode.parents.value[1])
        end,
         function ()
        if x.hasGrad x.grad .+= ifelse.(x.value .> zero(x.value), x.value, zero(x.value)) end
    end)
    return newNode
end



# --------------------------------------------------
#           other
# --------------------------------------------------

function sum(x::Node{<:AbstractArray})
    newNode = Node(sum(x.value), [x],function ()
            newNode.value = sum(newNode.parents.value[1])
        end,
         function ()
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



