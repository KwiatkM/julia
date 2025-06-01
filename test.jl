using Zygote

W = [1.0 2.0; 3.0 4.0]  # wagi 2x2
a = [0.5, 1.5]          # wektor wejściowy 2x1
b = [0.1, 0.2]          # bias 2x1

f(x) = sum((W * x + b) .* (W * x + b))

println(f(a))
println(f'(a))