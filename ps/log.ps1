param($p)

echo $p

if( $p -eq 'clean'){
    remove-item ./log/* 
}

if(-not (test-path ./log)){
    mkdir log
}

cmake --build ./build

$value = $LASTEXITCODE
$time = Get-Date -Format 'dd-hh-mm-ss'
$path = "./log/log" + $time + ".txt" 
echo "build exit code:" $value
if( 0 -eq $value){
    D:\bishe\gpu-database\build\Debug\gpu-database.exe > $path
    echo "exit code:" $LASTEXITCODE
}