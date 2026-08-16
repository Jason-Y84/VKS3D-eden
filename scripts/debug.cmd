@Echo OFF
SETlocal EnableExtensions
SETlocal EnableDelayedExpansion
pushd "%~dp0"

Set VKS3D_DIR=C:\Programs\VKS3D\Release
Set VKS3D_BUILDS=C:\Programs\VKS3D\Release\Builds\dev

For %%A in (%*) do (
    Set STEREO_LOGFILE_PATH=%~dp0%%~nxA\%%~nxA+VKS3D.log
    Set VKS3D_DUMP_SPIRV=%~dp0%%~nxA
    Set VKS3D_SKIP_SHADER_PATCHES=
    If exist "%%~A\*.dll" (
        copy /y "%%~A\*.dll" "!VKS3D_DIR!"
        If !ErrorLevel! NEQ 0 (
            Echo Failed to copy files. Error level: !ErrorLevel!
            Pause
        )
        exit
    ) else (
        If "%%~xA"==".zip" (
            for /f "tokens=2 delims=r@" %%R in ("%%~nA") do (
                If not exist "!VKS3D_BUILDS!\%%R" (MkDir "!VKS3D_BUILDS!\%%R")
                7z -y e "%%~A" -o"!VKS3D_BUILDS!\%%R" "-i^!*.dll"
                copy /y "!VKS3D_BUILDS!\%%R\*" "!VKS3D_DIR!"
                exit
                )
        ) else (
            If "%%~xA"==".spv" (
                Echo Validating "%%~nxA"
                If exist "!VULKAN_SDK!\Bin\spirv-val.exe" ("!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.0 "%%~A")
                If exist "!VULKAN_SDK!\Bin\spirv-val.exe" ("!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.1 "%%~A")
                If exist "!VULKAN_SDK!\Bin\spirv-val.exe" ("!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.2 "%%~A")
                If exist "!VULKAN_SDK!\Bin\spirv-val.exe" ("!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.3 "%%~A")
                If exist "!VULKAN_SDK!\Bin\spirv-val.exe" ("!VULKAN_SDK!\Bin\spirv-val.exe" "%%~A")
                If !ErrorLevel! NEQ 0 (Pause)
                Echo Decompiling "%%~nxA"
                If exist "!VULKAN_SDK!\Bin\spirv-dis.exe" ("!VULKAN_SDK!\Bin\spirv-dis.exe" "%%~A" -o "%%~dpA\%%~nxA.asm")

            ) else (
                pushD "%%~dpA"
                if not exist "%~dp0VKS3D\%%~nA" (
                    mkdir "%~dp0VKS3D\%%~nA"
                    ) else (
                    del /S /Q "%~dp0VKS3D\%%~nA\*.spv" 1>>NUL 2>&1
                    del /S /Q "%~dp0VKS3D\%%~nA\*.asm" 1>>NUL 2>&1
                    del /S /Q "%~dp0VKS3D\%%~nA\%%~nA+VKS3D.log" 1>>NUL 2>&1
                    If !ErrorLevel! NEQ 0 (
                        Echo Error deleting "%~dp0VKS3D\%%~nA\*"
                        Pause
                        )
                    )
                Echo %%~nA
                "%%~dpA%%~nxA"
                rem "%%~dpA%%~nxA" -fullscreen --fullscreen --width 1920 --height 1080
                If exist "%~dp0VKS3D\%%~nA\%%~nA+VKS3D.log" (Start "" "%~dp0VKS3D\%%~nA\%%~nA+VKS3D.log")
                If defined VKS3D_DUMP_SPIRV (
                    If exist "%~dp0VKS3D\%%~nA/*.spv" (
                        If exist "!VULKAN_SDK!\Bin\*.exe" (
                            pushD "%~dp0VKS3D\%%~nA"
                            For %%S in (*.spv) do (
                                If exist "!VULKAN_SDK!\Bin\spirv-val.exe" (
                                    Echo Validating "%%~nxS" reports:
                                    "!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.0 "%%~nxS"
                                    "!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.1 "%%~nxS"
                                    "!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.2 "%%~nxS"
                                    "!VULKAN_SDK!\Bin\spirv-val.exe" --target-env vulkan1.3 "%%~nxS"
                                    "!VULKAN_SDK!\Bin\spirv-val.exe" "%%~nxS"
                                    if !ErrorLevel! NEQ 0 (
                                        If exist "!VULKAN_SDK!\Bin\spirv-dis.exe" (
                                            Echo Decompiling "%%~nxS"...
                                            "!VULKAN_SDK!\Bin\spirv-dis.exe" "%%~nxS" -o "%%~nxS.asm"
                                        )
                                        Pause
                                    )
                                )
                                )
                            )
                        ) else (
                        If not exist "%~dp0VKS3D\%%~nA/*" (
                            rmdir /s /q "%~dp0VKS3D\%%~nA"
                            )
                        )
                    ) else (
                    If not exist "%~dp0VKS3D\%%~nA/*.log" (
                        If not exist "%~dp0VKS3D\%%~nA/*.spv" (
                            If not exist "%~dp0VKS3D\%%~nA/*.asm" (
                                rmdir /s /q "%~dp0VKS3D\%%~nA"
                                )
                            )
                        )
                    )
                Echo.
                )
            )
        )
    )
Echo Done.
pause
exit