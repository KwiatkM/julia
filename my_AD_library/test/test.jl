include("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\src\\MLP.jl")
using CSV
using DataFrames
using LinearAlgebra: normalize
using Random: shuffle
using Statistics: mean, std

df = CSV.read("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\Boston.csv", DataFrame)
data = Matrix(df)

X_train = convert(Matrix{Float64},data[:,2:14])
y_train = convert(Vector{Float64},data[:,15])

sh = shuffle(1:size(X_train,1))



for c in eachcol(X_train)
    m = mean(c)
    s = std(c)
    c .= (c .- m)./s
end

X_train = X_train'
X_train = X_train[: , sh]
y_train = y_train[ sh];
y_train = reshape(y_train,1,:)

model = buildModel(Float64,
    FullyConnected(size(X_train, 1),10,ReLU),
    FullyConnected(10,8,ReLU),
    FullyConnected(8,1,linear),
    Loss(myMSE)
    )
    
# epochs
# learning rate
options = Options(1000, 0.0001);

train(model, X_train, y_train, options)

println("Real value: ", y_train[1])
println("Inference value: ", inference(model,X_train[:,1]))
println("Real value: ", y_train[2])
println("Inference value: ", inference(model,X_train[:,2]))
println("Real value: ", y_train[3])
println("Inference value: ", inference(model,X_train[:,3]))
println("Real value: ", y_train[4])
println("Inference value: ", inference(model,X_train[:,4]))
println("Real value: ", y_train[5])
println("Inference value: ", inference(model,X_train[:,5]))