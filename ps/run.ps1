cmake --build ./build
$value = $LASTEXITCODE
echo "build exit code:" $value
if( 0 -eq $value){
    D:\bishe\gpu-database\build\Debug\gpu-database.exe
    echo "exit code:" $LASTEXITCODE
}