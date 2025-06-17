include("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\src\\MLP.jl")
using CSV
using DataFrames
using LinearAlgebra: normalize
using Random: shuffle
using Statistics: mean, std

# przygotowanie zbioru danych
df = CSV.read("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\Boston_lite.csv", DataFrame)
data = Matrix(df)

X_train = convert(Matrix{Float32},data[:,2:14])
y_train = convert(Vector{Float32},data[:,15])

sh = shuffle(1:size(X_train,1))

# normalizacja danych
for c in eachcol(X_train)
    c .= (c .- mean(c))./std(c)
end


X_train = X_train'
X_train = X_train[: , sh]
y_train = y_train[ sh];

X_train[4,:] .= 0

model = buildModel(Float64,
    FullyConnected(size(X_train, 1),10,ReLU),
    FullyConnected(10,8,ReLU),
    FullyConnected(8,1,linear),
    Loss(myMSE)
    )
    

# epochs
# learning rate
options = Options();#(100, 0.00001);
options.nEpochs = 10;
options.learningRate = 10e-5;
options.batchSize = 5;

train(model, X_train, y_train, options)

println("Real value: ", y_train[1])
println("Inference value: ", inference(model,X_train[:,1]))