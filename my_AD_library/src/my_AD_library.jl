
mutable struct Node{T}
    hasGrad :: Bool
    value :: T
    grad :: T 
    parents :: Vector{Node}
    forward::Function
    back :: Function
    nodeType :: String

    # hasGrad = true -> Variable
    # hasGard = false -> constant
    function Node(value::T, hasGrad::Bool=true) where T
        return new{T}(  hasGrad,
                        value,
                        zero(value),
                        Node[],
                        () -> nothing,
                        () -> nothing,
                        hasGrad ? "variable" : "constant")
    end

    # operator
    function Node(value::T, parents::Vector{<:Node},forward::Function, back::Function, info::String) where T
        return new{T}(  true,
                        value,
                        zero(value),
                        parents,
                        forward,
                        back,
                        info)
    end

end


Base.show(io::IO, p::Node) = print(io, p.nodeType, " (v=", size(p.value), ", grad=",size(p.grad),")")

# --------------------------------------------
# --------------------------------------------
#           Basic operations
# --------------------------------------------
# --------------------------------------------


import Base: +, -, *, /, sin, cos, tan, tanh, sum

# --------------------------------------------
#           Addition
# --------------------------------------------

function +(x::Node{T}, y::Node{T}) where T <: AbstractFloat
        newNode = Node(x.value + y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value + newNode.parents[2].value 
        end,
            function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad += newNode.grad end
        end, "addition")
    return newNode
end

function +(x::Node{<:AbstractArray{T}}, y::Node{<:AbstractArray{T}}) where T <: AbstractFloat
        newNode = Node(x.value + y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value + newNode.parents[2].value 
        end,
            function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad += newNode.grad end
    end, "addition")
    return newNode
end

# --------------------------------------------
#           Subtraction
# --------------------------------------------

function -(x::Node{T}, y::Node{T}) where T <: AbstractFloat
        newNode = Node(x.value - y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value - newNode.parents[2].value 
        end, 
        function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad -= newNode.grad end
    end, "subtraction")
    return newNode
end

function -(x::Node{<:AbstractArray{T}}, y::Node{<:AbstractArray{T}}) where T <: AbstractFloat
        newNode = Node(x.value - y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value - newNode.parents[2].value 
        end, 
        function ()
            if x.hasGrad x.grad += newNode.grad end
            if y.hasGrad y.grad -= newNode.grad end
    end, "subtraction")
    return newNode
end

function -(x::Node{T}) where T <: AbstractFloat
        newNode = Node(-x.value, [x],function ()
            newNode.value = -newNode.parents[1].value
        end,  function ()
            if x.hasGrad x.grad -= newNode.grad end

    end, "negation")
    return newNode
end

function -(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
        newNode = Node(-x.value, [x],function ()
            newNode.value = -newNode.parents[1].value
        end, function ()
            if x.hasGrad x.grad -= newNode.grad end

    end, "negation")
    return newNode
end

# --------------------------------------------------
#           Multiplication
# --------------------------------------------------

function *(x::Node{T}, y::Node{T}) where T <: AbstractFloat
    newNode = Node(x.value * y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value * newNode.parents[2].value 
        end, 
        function ()
        if x.hasGrad x.grad += y.value * newNode.grad end
        if y.hasGrad y.grad += x.value * newNode.grad end
    end, "multiplication")
    return newNode
end

function *(x::Node{<:AbstractArray{T}}, y::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(x.value * y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value * newNode.parents[2].value 
        end, 
        function ()
        if x.hasGrad x.grad += newNode.grad * y.value' end
        if y.hasGrad y.grad += x.value' * newNode.grad end
    end, "multiplication")
    return newNode
end


function Broadcast.broadcasted(::typeof(*), x::Node{<:AbstractArray{T}}, y::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(x.value .* y.value, [x,y],function ()
            newNode.value = newNode.parents[1].value .* newNode.parents[2].value 
        end, 
        function ()
        if x.hasGrad x.grad += newNode.grad .* y.value end
        if x.hasGrad y.grad += newNode.grad .* x.value end
    end, "multiplication elementwise")
    return newNode
end

# --------------------------------------------------
#           trigonometry
# --------------------------------------------------

function sin(x::Node{T}) where T <: AbstractFloat
    newNode = Node(sin(x.value), [x],function ()
            newNode.value = sin(newNode.parents[1].value)
        end,
        function ()
        if x.hasGrad x.grad += cos(x.value) * newNode.grad end
    end, "sin")
    return newNode
end

function sin(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(sin.(x.value), [x],function ()
            newNode.value = sin.(newNode.parents[1].value)
        end,
        function ()
        if x.hasGrad x.grad .+= cos.(x.value) .* newNode.grad end
    end, "sin")
    return newNode
end

function cos(x::Node{T}) where T <: AbstractFloat
    newNode = Node(cos(x.value), [x],function ()
            newNode.value = cos(newNode.parents[1].value)
        end,
        function ()
        if x.hasGrad x.grad -= sin(x.value) * newNode.grad end
    end, "cos")
    return newNode
end

function cos(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(cos.(x.value), [x],function ()
            newNode.value = cos.(newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad .-= sin.(x.value) .* newNode.grad end
    end, "cos")
    return newNode
end

function tan(x::Node{T}) where T <: AbstractFloat
    newNode = Node(tan(x.value), [x],function ()
            newNode.value = tan(newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad += 1 / (cos(x.value)^2) * newNode.grad end
    end, "tan")
    return newNode
end

function tan(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(tan.(x.value), [x],function ()
            newNode.value = tan.(newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad .+= 1 ./ (cos.(x.value).^2) .* newNode.grad end
    end, "tan")
    return newNode
end


# --------------------------------------------------
#           activation functions
# --------------------------------------------------

function tanh(x::Node{T}) where T <: AbstractFloat
    newNode = Node(tanh(x.value), [x],function ()
            newNode.value = tanh(newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad += (1 - newNode.value^2) * newNode.grad end
    end, "tanh")

    return newNode
end

function tanh(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(tanh.(x.value), [x],function ()
            newNode.value = tanh.(newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad .+= (1 .- newNode.value.^2) .* newNode.grad end
    end, "tanh")
    return newNode
end

function sigm(x::Node{T}) where T <: AbstractFloat
    newNode = Node(1/(1+ℯ^(-x.value)), [x],function ()
            newNode.value = 1/(1+ℯ^newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad += newNode.value * (1-newNode.value) * newNode.grad end
    end, "sigm")
    return newNode
end

function sigm(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(1 ./ (1 .+ ℯ.^(-x.value)), [x],function ()
            newNode.value = 1 ./(1 .+ℯ.^newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad .+= newNode.value .* (1 .-newNode.value) .* newNode.grad end
    end, "sigm")
    return newNode
end

function ReLU(x::Node{T}) where T <: AbstractFloat
    newNode = Node(max(0,x.value), [x],function ()
            newNode.value = max(0,newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad += x.value > zero(x.value) ? newNode.grad : zero(x.value) end
    end, "ReLU")
    return newNode
end

function ReLU(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(max.(0,x.value), [x],function ()
            newNode.value = max.(0,newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad .+= ifelse.(x.value .> zero(x.value), newNode.grad, zero(newNode.grad)) end
    end, "ReLU")
    return newNode
end

function linear(x::Node{T}) where T <: AbstractFloat
    newNode = Node(x.value, [x],function ()
            newNode.value = newNode.parents[1].value
        end,
         function ()
        if x.hasGrad x.grad += newNode.grad end
    end, "linear")
    return newNode
end

function linear(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(x.value, [x],function ()
            newNode.value = newNode.parents[1].value
        end,
         function ()
        if x.hasGrad x.grad .+= newNode.grad end
    end, "linear")
    return newNode
end


# --------------------------------------------------
#           other
# --------------------------------------------------

function sum(x::Node{<:AbstractArray{T}}) where T <: AbstractFloat
    newNode = Node(sum(x.value), [x],function ()
            newNode.value = sum(newNode.parents[1].value)
        end,
         function ()
        if x.hasGrad x.grad .+= newNode.grad end
    end, "sum")
    return newNode
end

function myBinaryCrossEntropy(y::Node{T}, ŷ::Node{T}) where T <: AbstractFloat
    newNode = Node(-(y.value * log(ŷ.value) + (1 - y.value)*log(1 - ŷ.value)), [y,ŷ],function ()
            newNode.value = -(newNode.parents[1].value * log(newNode.parents[2].value) + (1 - newNode.parents[1].value)*log(1 - newNode.parents[1].value))
        end,
         function ()
        if ŷ.hasGrad ŷ.grad -= (y.value/ŷ.value - ((1-y.value)/(1-ŷ.value))) end
    end, "BCE")
    return newNode    
end

function myBinaryCrossEntropy(y::Node{T}, ŷ::Node{<:AbstractVector}) where T <: AbstractFloat
    newNode = Node(-(y.value * log(ŷ.value[1]) + (1 - y.value)*log(1 - ŷ.value[1])), [y,ŷ],function ()
            newNode.value = -(newNode.parents[1].value * log(newNode.parents[2].value[1]) + (1 - newNode.parents[1].value)*log(1 - newNode.parents[2].value[1]))
        end,
         function ()
        if ŷ.hasGrad ŷ.grad[1] -= (y.value / ŷ.value[1]) - ((1.f0 - y.value)/(1.f0 - ŷ.value[1])) end #ŷ.value[1] - y.value end
    end, "BCE")
    return newNode    
end

function myMSE(y::Node{T}, ŷ::Node{T}) where T <: AbstractFloat
    newNode = Node((y.value - ŷ.value)^2, [y,ŷ],function ()
            newNode.value = (newNode.parents[1].value - newNode.parents[2].value)^2
        end,
         function ()
        if ŷ.hasGrad ŷ.grad += 2*ŷ.value - 2*y.value end #ŷ.value[1] - y.value end
    end, "MSE")
    return newNode    
end

function myMSE(y::Node{T}, ŷ::Node{<:AbstractVector}) where T <: AbstractFloat
    newNode = Node((y.value - ŷ.value[1])^2, [y,ŷ],function ()
            newNode.value = (newNode.parents[1].value - newNode.parents[2].value[1])^2
        end,
         function ()
        if ŷ.hasGrad ŷ.grad[1] += 2*ŷ.value[1] - 2*y.value end
    end, "MSE")
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

function backward(head::Node{Vector{T}},order::Vector{Node}, seed::Vector{T}) where {T}
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
        backward(head,order,ones(typeof(head.value[1]),size(head.value)))# default seed for arrays and matrices
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

function forward(head::Node)
    for p in head.parents
        forward(p)
    end
    head.forward()
end



