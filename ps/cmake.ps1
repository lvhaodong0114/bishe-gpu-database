if(-not (test-path ./build)){
    mkdir build
}

remove-item ./build/* -Recurse

cd ./build
cmake ..
cd ..