using JLD2

X_train = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "X_train")
y_train = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "y_train")
X_test = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "X_test")
y_test = load("C:\\Users\\kwiat\\projekty\\julia\\my_AD_library\\data\\imdb_dataset_prepared.jld2", "y_test")

println(size(X_train))
println(size(y_train))
println(y_train[1:20])