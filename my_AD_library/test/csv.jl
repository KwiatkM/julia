include("../src/MLP.jl")

using CSV
using DataFrames

df = CSV.read("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\Boston.csv", DataFrame)

data = Matrix(df)

X_train = data[:,1:14]
y_train = data[:,15]

f,W,d = buildModel(
    FullyConnected(size(X_train, 2),10,ReLU),
    FullyConnected(10,1,ReLU),
    )

model = Model(f,W,d,
    binaryAccuracy)

# println(model.weightNodes)
# train(model, X_train,y_train*1f0,5)