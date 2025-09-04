include("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\src\\CNN.jl")

using JLD2

X_train = load("C:\\Users\\kwiat\\projekty\\julia\\CNN\\data\\imdb_dataset_prepared.jld2", "X_test") 
embeddings = load("C:\\Users\\kwiat\\projekty\\julia\\CNN\\data\\imdb_dataset_prepared.jld2", "embeddings")

x = Node(X_train[:,1:2])
m = dataMap(x,embeddings)
p = myPermutedims(m,(2,1,3))
c = im2col(p,3)
K = Node(randn(Float32,8,150))
mul = K * c
b = Node(rand(Float32,8,1))
mul_b = MLPaddBias(mul,b)
r = ReLU(mul_b)
r_re = convResultReshape(r,128) # TODO: policzyć tą drugą wartość
po = max_pool_1d(r_re,8)
f = flatten(po)

order = topo_sort(f)

# backward(f,order)
