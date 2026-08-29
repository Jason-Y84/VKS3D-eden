# Build and deploy VKS3D DLL
$ErrorActionPreference = "Continue"

# Import VS dev shell
$vcvars = "D:\Program Files (x86)\VisualStudio\18\community\VC\Auxiliary\Build\vcvars64.bat"
$cmake = "D:\Program Files (x86)\VisualStudio\18\community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$buildDir = "D:\Program Files (x86)\Trae CN\new project\VKS3D-stable\out\build\x64-Release"
$src = "D:\Program Files (x86)\Trae CN\new project\VKS3D-stable\out\build\x64-Release\Release\VKS3D_x64.dll"
$dst = "D:\Program Files (x86)\Trae CN\new project\VKS3D-stable\scripts\VKS3D_x64.dll"

# Build
Set-Location $buildDir
& $cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BUILD FAILED] cmake build returned non-zero exit code $LASTEXITCODE"
    exit 1
}

# Deploy
Copy-Item -Path $src -Destination $dst -Force
if (-not (Test-Path $dst)) {
    Write-Host "[DEPLOY FAILED] DLL not found at destination"
    exit 1
}

Write-Host "[BUILD OK] VKS3D_x64.dll deployed to scripts\"
Write-Host "Source: $src"
Write-Host "Destination: $dst"
Write-Host "LastWriteTime: $((Get-Item $dst).LastWriteTime)"
