include("../src/my_AD_library.jl")

using Zygote

# scalar tests

# test 1.1: sin(x^2), x = 5
x1 = Node(5., true)
f1 = sin(x1 * x1)
order1 = topo_sort(f1)
backward(f1,order1)

g1(x) = sin(x*x)

println("--- test 1.1 ---")
println("my_library: value=", f1.value, ", grad=", x1.grad)
println("    zygote: value=", g1(5.), ", grad=", g1'(5.))


# test 1.2: cos(x-2) + sin(x^x) - tan(2x), x = 6.2
x2 = Node(6.2, true)
c2 = Node(2., false)
f2 = cos(x2 - c2) + sin(x2*x2) - tan(c2 * x2)
order2 = topo_sort(f2)
backward(f2,order2)

g2(x) = cos(x-2.) + sin(x*x) - tan(2x)

println("--- test 1.2 ---")
println("my_library: value=", f2.value, ", grad=", x2.grad)
println("    zygote: value=", g2(6.2), ", grad=", g2'(6.2))

# test 1.3: tanh(-x), x = 4.1
x3 = Node(4.1, true)
f3 = tanh(-x3)
order3 = topo_sort(f3)
backward(f3,order3)

g3(x) = tanh(-x)

println("--- test 1.3 ---")
println("my_library: value=", f3.value, ", grad=", x3.grad)
println("    zygote: value=", g3(4.1), ", grad=", g3'(4.1))


# matrix tests

# test 2.1: ∑(W * x + b)
W4 = Node([1. 2.; 3. 4.],true)
x4 = Node([0.5, 1.5],true)         
b4 = Node([0.1, 0.2],true)          

f4 = sum(W4 * x4 + b4)
order4 = topo_sort(f4)
backward(f4,order4)

W_ = [1. 2.; 3. 4.]
b_ = [0.1, 0.2]
x_ = [0.5, 1.5]
g4(X) = sum(W_ * X + b_)

println("--- test 2.1 ---")
println("my_library: value=", f4.value, ", grad=", x4.grad)
println("    zygote: value=", g4(x_), ", grad=", g4'(x_))

# test 2.2: ∑(tanh(W * x))
W5 = Node([1. 2.; 3. 4.],true)
x5 = Node([0.5, 1.5],false)      

f5 = sum(tanh(W5 * x5))
order5 = topo_sort(f5)
backward(f5,order5)

W_ = [1. 2.; 3. 4.]
x_ = [0.5, 1.5]
g5(W) = sum(tanh.(W * x_))

println("--- test 2.2 ---")
println("my_library: value=", f5.value, ", grad=", W5.grad)
println("    zygote: value=", g5(W_), ", grad=", g5'(W_))

# test 2.3: activation functions

tmp = randn(1,5)
x6 = Node(tmp,true)   
x7 = Node(tmp,true) 
x8 = Node(tmp,true)   

f6 = ReLU(x6)
f7 = tanh(x7)
f8 = sigm(x8)

order6 = topo_sort(f6)
order7 = topo_sort(f7)
order8 = topo_sort(f8)
backward(f6,order6)
backward(f7,order7)
backward(f8,order8)


println("--- test 2.3 ---")
println("original vector:")
println(tmp)
println("ReLU")
println("   value: ",f6.value)
println("    grad: ",x6.grad)
println("tanh")
println("   value: ",f7.value)
println("    grad: ",x7.grad)
println("sigm")
println("   value: ",f8.value)
println("    grad: ",x8.grad)