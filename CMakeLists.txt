cmake_minimum_required(VERSION 3.20)

# ── Version number defaults ─────────────────────────────────────────────────
# Must be set BEFORE project() because the VERSION token is expanded by
# project() itself.  VS 2026 ships a more recent CMake that strictly
# validates VERSION strings and rejects "1.0.." (trailing dots caused by
# unset VKS3D_VER_PATCH / VKS3D_VER_BUILD at project evaluation time) as
# well as any component that contains non-digit characters, whitespace,
# leading empty segments, or line-ending artefacts that sneak into the
# command line (GitHub Actions pwsh backtick-continuation lines have
# historically produced exactly this class of malformed flags on the
# VKS3D self-hosted runner pool at D:\a — so we kill the build with a
# sharp FATAL_ERROR *before* project() instead of letting the obscure
# "VERSION format invalid" one-liner bubble up from inside cmProjectCommand).
#
# Can still be overridden on the command line:
#   cmake .. -DVKS3D_VER_PATCH=10 -DVKS3D_VER_BUILD=2
if(NOT DEFINED VKS3D_VER_PATCH)
    set(VKS3D_VER_PATCH 0 CACHE STRING "Patch version number (0-65535)")
endif()
if(NOT DEFINED VKS3D_VER_BUILD)
    set(VKS3D_VER_BUILD 0 CACHE STRING "Build version number (0-65535)")
endif()

# ── Defensive VERSION-component validation (fail before project()).
foreach(_ver_comp IN ITEMS VKS3D_VER_PATCH VKS3D_VER_BUILD)
    set(_val "${${_ver_comp}}")
    if("${_val}" STREQUAL "")
        message(FATAL_ERROR
            "${_ver_comp} is empty.  CMake project(VERSION 1.0.P.B) requires "
            "four integer segments; the current value would produce a "
            "double-dot such as VERSION 1.0..0 which CMake 3.28+ rejects. "
            "Pass -D${_ver_comp}=<uint16> on the cmake command line.")
    endif()
    if(NOT "${_val}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "${_ver_comp}='${_val}' is not a plain non-negative integer.  "
            "Only ASCII digits 0..9 are allowed inside a VERSION segment; "
            "whitespace, CRLF, dollar signs, and shell quoting artefacts "
            "(common when GitHub Actions pwsh expands env vars across "
            "backtick-continuation lines) cause exactly this failure. "
            "Source of the value: command-line -> CACHE -> default(0).")
    endif()
    if(${_val} GREATER 65535)
        message(FATAL_ERROR
            "${_ver_comp}=${_val} exceeds 16-bit ceiling 65535 used by the "
            "VS_VERSION_INFO .rc resource stamp (FILEVERSION/PRODUCTVERSION "
            "are each 4x uint16).  VKS3D_VER_PATCH comes from GitHub's "
            "GITHUB_RUN_NUMBER which will not wrap for ~18 centuries of "
            "daily pushes; if you see this, something injected an "
            "out-of-range string (hex, overflow, the string name of the "
            "environment variable itself instead of its contents).")
    endif()
endforeach()

project(VKS3D
    VERSION 1.0.${VKS3D_VER_PATCH}.${VKS3D_VER_BUILD}
    LANGUAGES C CXX
    DESCRIPTION "Vulkan Stereoscopic 3D ICD"
)

message(STATUS "VKS3D: version ${PROJECT_VERSION}")

# ── Policy: MSVC requires C11 via /std:c11 ─────────────────────────────────
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Policy: present_nv3d.cpp requires C20 via /std:c++20 ─────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Determine bitness from generator/toolchain ──────────────────────────────
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(STEREO_BITS 64)
    set(STEREO_ARCH "x64")
else()
    set(STEREO_BITS 32)
    set(STEREO_ARCH "x86")
endif()
message(STATUS "VKS3D: building ${STEREO_ARCH} (${STEREO_BITS}-bit)")

# ── Output names ────────────────────────────────────────────────────────────
#   VKS3D_x64.dll  /  VKS3D_x86.dll   (Windows)
#   VKS3D_x64.so   /  VKS3D_x86.so    (Linux)
set(TARGET_NAME VKS3D_${STEREO_ARCH})

# ── Configure RC file from template ──────────────────────────────────────────
configure_file(
    "${CMAKE_SOURCE_DIR}/src/VKS3D.rc.in"
    "${CMAKE_BINARY_DIR}/VKS3D.rc"
)

# ── Vulkan SDK ───────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# Also try the VulkanSDK env var explicitly (MSVC installs don't always set
# the cmake config path, but the env var is always present after SDK install)
if(NOT Vulkan_FOUND AND DEFINED ENV{VULKAN_SDK})
    set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/Include")
    message(STATUS "VKS3D: Using Vulkan headers from VULKAN_SDK env: ${Vulkan_INCLUDE_DIR}")
endif()

# ── Source files ─────────────────────────────────────────────────────────────
# NOTE: if you add / rename / remove a source file here, you MUST re-run
# `cmake -S . -B build` from scratch.  CI self-hosted runners and stale
# build dirs have been observed to keep using a cached vcxproj that does
# not list the newly added TU — producing LNK2019 for every symbol it
# defines (see: stereo_CmdBlitImage / stereo_CmdCopyImage / …).
set(STEREO_SOURCES
    src/icd_main.c
    src/stereo.c
    src/instance.c
    src/device.c
    src/physdev.c
    src/physdev_ext.c
    src/render_pass.c
    src/shader.c
    src/swapchain.c
    src/present.c
    src/present_alt.c
    src/framebuffer.c
    src/dxgi_output.c
    src/ini.c
    src/tes_inject.c
    src/image_blit.c
)

# ── Fail-fast: every listed source must actually exist on disk.
# Catches (a) forgotten `git add` on a freshly added TU on GitHub CI,
# (b) partial submodule checkouts, (c) case-sensitive filesystem typos,
# ALL of which otherwise surface later as an opaque LNK2019.
foreach(_src IN LISTS STEREO_SOURCES)
    set(_abs "${CMAKE_SOURCE_DIR}/${_src}")
    if(NOT EXISTS "${_abs}")
        message(FATAL_ERROR
            "VKS3D: SOURCE FILE MISSING — ${_abs}\n"
            "  This usually means:\n"
            "    1. `git add` was not run for the new source file before push,\n"
            "    2. `.gitattributes` / LFS pointer replaced the real content,\n"
            "    3. CI checkout forgot `submodules: recursive` / a patch.\n"
            "  Fix the repo contents and re-run `cmake` from an EMPTY build dir.")
    endif()
endforeach()
message(STATUS "VKS3D: ${STEREO_SOURCES} sources verified on disk (fail-fast OK)")

# ── Optional third-party dependencies ---------------------------------------
#
# NV3D-Lib (NVIDIA 3D Vision direct-mode output) and SPIRV-Headers are
# pulled in via git submodules.  When the user clones without
# --recurse-submodules (the most common situation), the directories are
# empty and linking fails.
#
# Neither dependency is required for the default SBS / TAB / INTERLACED /
# DXGI presentation modes used by Yuzu and every Vulkan game.  Only the
# dedicated "nv3dlib" mode needs NV3D-Lib, and it is disabled at runtime
# if the DLL failed to initialise anyway.
#
# So we:
#   * Detect whether each submodule has actually been checked out.
#   * If not, skip the include / link / compile-def for it.
#   * Add a preprocessor define so present_nv3d.cpp can compile to a
#     no-op stub even when NV3D headers are absent.
set(VKS3D_NV3DLIB_DIR     "${CMAKE_SOURCE_DIR}/third_party/NV3D-Lib")
set(VKS3D_NV3DLIB_LIB_x64 "${VKS3D_NV3DLIB_DIR}/NV3DLib/lib/NV3DLib.lib")
set(VKS3D_NV3DLIB_LIB_x86 "${VKS3D_NV3DLIB_DIR}/NV3DLib/lib/NV3DLib.lib")
set(VKS3D_NVAPI_LIB_x64   "${VKS3D_NV3DLIB_DIR}/external/nvapi/amd64/nvapi64.lib")
set(VKS3D_NVAPI_LIB_x86   "${VKS3D_NV3DLIB_DIR}/external/nvapi/x86/nvapi.lib")
set(VKS3D_SPIRV_HEADERS   "${CMAKE_SOURCE_DIR}/third_party/SPIRV-Headers/include")

# Detection must key on NV3DLib.lib itself, NOT on the nvapi import
# lib.  The upstream oneup03/NV3D-Lib submodule tracks
# external/nvapi/... but does NOT track NV3DLib/lib/NV3DLib.lib (that
# binary only exists in some local checkouts), so an `EXISTS(nvapi) OR
# EXISTS(NV3DLib)` test reports HAVE=ON on CI and then dies at link
# time with LNK1181 "cannot open input file NV3DLib.lib".  Requiring
# the actual link input makes the stub fallback (present.c weak-impl
# -> nv3d_* return false) kick in instead — which is exactly the
# configuration Yuzu SBS mode already runs with.
set(VKS3D_HAVE_NV3DLIB    OFF)
if(EXISTS "${VKS3D_NV3DLIB_LIB_x64}")
    set(VKS3D_HAVE_NV3DLIB ON)
endif()
set(VKS3D_HAVE_SPIRV_HEADERS OFF)
if(IS_DIRECTORY "${VKS3D_SPIRV_HEADERS}")
    set(VKS3D_HAVE_SPIRV_HEADERS ON)
endif()

message(STATUS "VKS3D: NV3D-Lib     = ${VKS3D_HAVE_NV3DLIB}  (skipped — Yuzu SBS mode does not need it)")
message(STATUS "VKS3D: SPIRV-Headers = ${VKS3D_HAVE_SPIRV_HEADERS}")

# Only compile present_nv3d.cpp when the submodule is actually present.
# Otherwise stubs in stereo_icd.h (nv3d_init/nv3d_destroy/nv3d_present
# returning false) will be linked via the weak-impl pattern in present.c.
if(VKS3D_HAVE_NV3DLIB)
    list(APPEND STEREO_SOURCES src/present_nv3d.cpp)
endif()

# ── Shader compilation ───────────────────────────────────────────────────────
find_program(GLSLC     glslc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
find_program(GLSLANG   glslangValidator
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")

if(GLSLC OR GLSLANG)
    set(SHADER_DIR   "${CMAKE_SOURCE_DIR}/shaders")
    set(SPIRV_OUT    "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${SPIRV_OUT}")

    set(SHADER_TARGETS "")

    # Each entry is "stage:basename" — glslc needs explicit -fshader-stage=
    # because it does not infer the stage from a .glsl extension.
    set(SHADERS "vert:composite.vert" "frag:composite.frag")

    foreach(SHADER_ENTRY ${SHADERS})
        string(REPLACE ":" ";" SHADER_PARTS "${SHADER_ENTRY}")
        list(GET SHADER_PARTS 0 STAGE)
        list(GET SHADER_PARTS 1 SHADER)

        set(GLSL_FILE  "${SHADER_DIR}/${SHADER}.glsl")
        set(SPIRV_FILE "${SPIRV_OUT}/${SHADER}.spv")

        if(GLSLC)
            add_custom_command(OUTPUT "${SPIRV_FILE}"
                COMMAND "${GLSLC}" -fshader-stage=${STAGE} "${GLSL_FILE}" -o "${SPIRV_FILE}"
                DEPENDS "${GLSL_FILE}"
                COMMENT "Compiling ${SHADER}.glsl -> ${SHADER}.spv")
        else()
            add_custom_command(OUTPUT "${SPIRV_FILE}"
                COMMAND "${GLSLANG}" -V -S "${STAGE}" "${GLSL_FILE}" -o "${SPIRV_FILE}"
                DEPENDS "${GLSL_FILE}"
                COMMENT "Compiling ${SHADER}.glsl -> ${SHADER}.spv (glslangValidator)")
        endif()

        list(APPEND SHADER_TARGETS "${SPIRV_FILE}")
    endforeach()

    add_custom_target(shaders ALL DEPENDS ${SHADER_TARGETS})
else()
    message(WARNING "Neither glslc nor glslangValidator found — shaders won't be compiled.\n"
                    "Install the Vulkan SDK or set VULKAN_SDK.")
endif()

# ── DLL / shared library ─────────────────────────────────────────────────────
add_library(${TARGET_NAME} SHARED ${STEREO_SOURCES})

set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME  ${TARGET_NAME}
    PREFIX       ""           # no "lib" prefix on Linux either
    C_VISIBILITY_PRESET hidden
)

# Vulkan SDK ships SPIR-V headers under <spirv-headers/spirv.h>, which
# covers every Spv* token used by shader.c without the SPIRV-Headers
# git submodule.  The submodule path is kept for users who have it.
target_include_directories(${TARGET_NAME} PRIVATE
    include
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/third_party/SPIRV-Headers/include
)
if(VKS3D_HAVE_NV3DLIB)
    target_include_directories(${TARGET_NAME} PRIVATE
        third_party/NV3D-Lib/NV3DLib/include)
endif()

# ── Version string: embed git commit hash + build date ───────────────────────
# Baked in at configure time so it appears in the log without any runtime
# calls to git.  Falls back to "unknown" when building outside a git repo.
find_program(GIT_EXECUTABLE git)
set(VKS3D_GIT_COMMIT "unknown")
if(GIT_EXECUTABLE)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VKS3D_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
string(TIMESTAMP VKS3D_BUILD_DATE "%Y-%m-%d" UTC)
message(STATUS "VKS3D: commit=${VKS3D_GIT_COMMIT}  date=${VKS3D_BUILD_DATE}")

target_compile_definitions(${TARGET_NAME} PRIVATE
    VKS3D_GIT_COMMIT="${VKS3D_GIT_COMMIT}"
    VKS3D_BUILD_DATE="${VKS3D_BUILD_DATE}"
    VKS3D_VERSION="${PROJECT_VERSION}"
)

# ── Windows-specific settings ────────────────────────────────────────────────
if(WIN32)
    target_sources(${TARGET_NAME} PRIVATE src/VKS3D.def "${CMAKE_BINARY_DIR}/VKS3D.rc")
# advapi32 for RegOpenKeyEx / RegEnumValue (registry ICD detection)
target_link_libraries(${TARGET_NAME} PRIVATE advapi32)
target_link_libraries(${TARGET_NAME} PRIVATE
    dxgi.lib
    d3d11.lib
    d3d9.lib
)
    if(VKS3D_HAVE_NV3DLIB)
        target_link_libraries(${TARGET_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/third_party/NV3D-Lib/NV3DLib/lib/NV3DLib.lib)
        if(STEREO_BITS EQUAL 64)
            target_link_libraries(${TARGET_NAME} PRIVATE
                ${CMAKE_SOURCE_DIR}/third_party/NV3D-Lib/external/nvapi/amd64/nvapi64.lib
            )
        else()
            target_link_libraries(${TARGET_NAME} PRIVATE
                ${CMAKE_SOURCE_DIR}/third_party/NV3D-Lib/external/nvapi/x86/nvapi.lib
            )
        endif()
    endif()

    target_compile_definitions(${TARGET_NAME} PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        VKS3D_ARCH="${STEREO_ARCH}"
        VKS3D_HAVE_NV3DLIB=$<IF:$<BOOL:${VKS3D_HAVE_NV3DLIB}>,1,0>
        VKS3D_HAVE_SPIRV_HEADERS=$<IF:$<BOOL:${VKS3D_HAVE_SPIRV_HEADERS}>,1,0>
    )
    # MSVC: enable C11, treat warnings sensibly, force UTF-8 source charset
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            /std:c11
            /W3
            /utf-8      # source and execution charset = UTF-8 (fixes C4819 for non-ASCII comments)
            /wd4100     # unreferenced formal parameter
            /wd4152     # nonstandard extension: function/data pointer conversion
        )
    else()
        # MinGW / clang-cl
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wno-unused-parameter
        )
    endif()

    # Install DLL + JSON manifest together
    install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "."
    )
    install(FILES "manifest/VKS3D_${STEREO_ARCH}.json"
        DESTINATION "."
    )
    install(FILES "scripts/install.bat" "scripts/uninstall.bat" "scripts/Install-VKS3D.ps1"
        DESTINATION "."
    )

else()
    # ── Linux ────────────────────────────────────────────────────────────────
    target_link_libraries(${TARGET_NAME} PRIVATE dl pthread m)

    # Linker version script: only expose the three ICD symbols
    if(EXISTS "${CMAKE_SOURCE_DIR}/stereo_icd.map.in")
        configure_file(stereo_icd.map.in stereo_icd.map @ONLY)
        set_target_properties(${TARGET_NAME} PROPERTIES
            LINK_FLAGS "-Wl,--version-script=${CMAKE_BINARY_DIR}/stereo_icd.map"
        )
    endif()

    install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION lib)
    install(FILES "manifest/VKS3D_${STEREO_ARCH}.json" DESTINATION share/vulkan/icd.d)
endif()

if(TARGET shaders)
    add_dependencies(${TARGET_NAME} shaders)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
option(VKS3D_BUILD_TESTS "Build unit tests" OFF)
option(VKS3D_BUILD_DEMO  "Build integration demo" OFF)

# Diagnostics tool is always built — it checks all prerequisites for stereo
# rendering and is useful even in release builds / installed alongside the DLL.
# enable_testing()
# add_subdirectory(tests)

# ── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "  VKS3D build summary")
message(STATUS "  -------------------")
message(STATUS "  Target     : ${TARGET_NAME}")
message(STATUS "  Arch       : ${STEREO_ARCH} (${STEREO_BITS}-bit)")
message(STATUS "  Vulkan SDK : ${Vulkan_INCLUDE_DIRS}")
# message(STATUS "  Tests      : ${VKS3D_BUILD_TESTS}")
message(STATUS "")
