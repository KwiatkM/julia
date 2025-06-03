include("../src/MLP.jl")

using JLD2
# using CSV
# using DataFrames

X_train = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "X_train")
y_train = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "y_train")
X_test = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "X_test")
y_test = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "y_test")

# df = CSV.read("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\Boston.csv", DataFrame)

# data = Matrix(df)

# println(size(X_test))
# println(size(y_test))
# println(y_train[1:20])
# println(X_train[1:5,1:5])
# println(typeof(y_train))
# println(typeof(X_train))


# using CSV, DataFrames
# data = CSV.read()


f,W,d,y = buildModel(
    FullyConnected(size(X_train, 1),32,ReLU),
    FullyConnected(32,1,sigm),
    Loss(myBinaryCrossEntropy))

model = Model(f,W,d,y,
    binaryAccuracy)

train(model, X_train,y_train*1f0,5)


using Flux

# y = Node(1.f0)
# ŷ = Node([0.93f0])
# test = myBinaryCrossEntropy(y, ŷ)
# order = topo_sort(test)
# backward(test, order)

# println(test.value)
# println(ŷ.grad)

# forward(test)
# println(test.value)
# println(ŷ.grad)

# println(Flux.binarycrossentropy(0.93f0,1.f0))
# println(gradient(Flux.binarycrossentropy, 0.93f0,1.f0))

# y = Node(1.f0)
# ŷ = Node([0.93f0])
# test = myMSE(y, ŷ)
# order = topo_sort(test)
# backward(test, order)

# println(test.value)
# println(ŷ.grad)

# forward(test)
# println(test.value)
# println(ŷ.grad)

# println(Flux.mse(0.93f0,1.f0))
# println(gradient(Flux.mse, 0.93f0,1.f0))

# y = Node([0.54f0, 1.35f0, -3.51f0])
# test = ReLU(ReLU(sigm(y))+y)

# order = topo_sort(test)
# backward(test, order)

# println(test.value)
# println(y.grad)


# m(x) = NNlib.relu(NNlib.relu(NNlib.sigmoid(x))+x)



# println(m([0.54f0, 1.35f0, -3.51f0]))
# println(Flux.jacobian(m,[0.54f0, 1.35f0, -3.51f0]))
# Flux.gradient

