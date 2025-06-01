include("../src/my_AD_library.jl")



W = Node([1.0 2.0; 3.0 4.0],true)  # wagi 2x2
x = Node([0.5, 1.5],false)          # wektor wejściowy 2x1
b = Node([0.1, 0.2],true)          # bias 2x1

y = W * x + b

loss = sum(mult_ew(y,y))

order = topo_sort(loss)
backward(loss,order)

println("W grad = ", W.grad)
backward(loss,order)
println("W grad = ", W.grad)
# println("\nend of test")