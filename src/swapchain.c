/*
 * swapchain.c — GPU blit SBS stereo swapchain
 *
 * External-memory + multiview architecture (DXGI path) plus
 * GPU-blit compose path (SBS/TAB/Interlaced) replacing the old CPU readback.
 *
 * Compose path frame flow:
 *   App renders → stereo_images[0] (layers 0=left, 1=right)
 *   QueuePresentKHR → gpu_compose_present:
 *     AcquireNextImageKHR (real SC) → barrier → CmdBlitImage × 2 → Present
 *   AcquireNextImageKHR (fake) → WaitForFences(barrier_fences[0])
 *     ensures stereo_images[0] is safe for next render
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stereo_icd.h"
#include "dxgi_output.h"
#include "present_alt.h"
#include "present_nv3d.h"

/* ── Helper: find suitable memory type ──────────────────────────────────── */
static uint32_t find_memory_type(StereoDevice *sd, uint32_t type_bits,
                                  VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    sd->si->real.GetPhysicalDeviceMemoryProperties(sd->real_physdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

static void tracked_destroy_image(
    StereoDevice *sd,
    VkImage image,
    const char *site);

/* ── Allocate VkImage backed by imported D3D11 NT-handle memory ─────────── */
static VkResult alloc_external_stereo_image(StereoDevice *sd, StereoSwapchain *sc,
                                             VkImage *out_image, VkDeviceMemory *out_mem)
{
    VkExternalMemoryImageCreateInfo ext_img = {
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
    };
    VkImageCreateInfo ici = {
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = &ext_img,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = sc->format,
        .extent        = {sc->app_width, sc->app_height, 1},
        .mipLevels     = 1,
        .arrayLayers   = 2,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    STEREO_LOG_VERBOSE("CALL real CreateImage");
    VkResult res = sd->real.CreateImage(sd->real_device, &ici, NULL, out_image);
    STEREO_LOG_VERBOSE("RETURN real CreateImage result=%d", res);
    if (res != VK_SUCCESS) { STEREO_ERR("CreateImage(external) failed: %d", res); return res; }

    VkMemoryRequirements mr;
    sd->real.GetImageMemoryRequirements(sd->real_device, *out_image, &mr);
    if (!sd->real.GetMemoryWin32HandlePropertiesKHR) {
        STEREO_ERR("GetMemoryWin32HandlePropertiesKHR not loaded");
        tracked_destroy_image(
            sd,
            *out_image,
            "external image failure");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    VkMemoryWin32HandlePropertiesKHR hp = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
    res = sd->real.GetMemoryWin32HandlePropertiesKHR(sd->real_device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, sc->shared_nt_handle, &hp);
    if (res != VK_SUCCESS) {
        STEREO_ERR("GetMemoryWin32HandlePropertiesKHR failed: %d", res);
        tracked_destroy_image(
            sd,
            *out_image,
            "external image failure"); return res;
    }
    uint32_t mt = find_memory_type(sd, mr.memoryTypeBits & hp.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) {
        STEREO_ERR("No compatible memory for external image");
        tracked_destroy_image(
            sd,
            *out_image,
            "external image failure");
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    VkImportMemoryWin32HandleInfoKHR import_info = {
        .sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
        .handle     = sc->shared_nt_handle,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import_info,
        .allocationSize = mr.size, .memoryTypeIndex = mt,
    };
    res = sd->real.AllocateMemory(sd->real_device, &mai, NULL, out_mem);
    if (res != VK_SUCCESS) {
        STEREO_ERR("AllocateMemory(import) failed: %d", res);
        tracked_destroy_image(
            sd,
            *out_image,
            "external image failure"); return res;
    }
    sc->shared_nt_handle = NULL;
    res = sd->real.BindImageMemory(sd->real_device, *out_image, *out_mem, 0);
    if (res != VK_SUCCESS) {
        STEREO_ERR("BindImageMemory(external) failed: %d", res);
        sd->real.FreeMemory(sd->real_device, *out_mem, NULL);
        tracked_destroy_image(
            sd,
            *out_image,
            "external image failure");
    }
    return res;
}

/* ── Barrier CB + fence for frame sync ──────────────────────────────────── */
static bool setup_barrier_resources(StereoDevice *sd, StereoSwapchain *sc)
{
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = sd->gfx_qf,
    };
    if (sd->real.CreateCommandPool(sd->real_device, &cpci, NULL, &sc->barrier_pool)
            != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = sc->barrier_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (sd->real.AllocateCommandBuffers(sd->real_device, &cbai, sc->barrier_cmds)
            != VK_SUCCESS) return false;
    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,  /* starts signaled → first acquire never blocks */
    };
    return sd->real.CreateFence(sd->real_device, &fci, NULL, &sc->barrier_fences[0])
           == VK_SUCCESS;
}

/* ── Allocate stereo render target (2-layer image + view, no CPU staging) ─ */
static VkResult alloc_alt_stereo_swapchain(StereoDevice *sd, StereoSwapchain *sc)
{
    sc->image_count      = 1;
    sc->stereo_images    = calloc(1, sizeof(VkImage));
    sc->stereo_memory    = calloc(1, sizeof(VkDeviceMemory));
    sc->stereo_views_arr = calloc(1, sizeof(VkImageView));
    sc->barrier_cmds     = calloc(1, sizeof(VkCommandBuffer));
    sc->barrier_fences   = calloc(1, sizeof(VkFence));
    if (!sc->stereo_images || !sc->stereo_memory || !sc->stereo_views_arr ||
        !sc->barrier_cmds  || !sc->barrier_fences)
        return VK_ERROR_OUT_OF_HOST_MEMORY;

    VkResult res = alt_alloc_stereo_image(sd, sc,
                       &sc->stereo_images[0], &sc->stereo_memory[0]);
    if (res != VK_SUCCESS) return res;

    extern void check_array_count(uint32_t *, uint32_t, const char *);
    CHECK_ARRAY_COUNT(sd->intercepted_color_count, MAX_COLOR_IMAGES, "intercepted_color_count");
    bool already_tracked = false;
    for (uint32_t i = 0; i < sd->intercepted_color_count; i++)
    {
        if (sd->intercepted_color[i] == sc->stereo_images[0])
        {
            already_tracked = true;
            break;
        }
    }
    if (!already_tracked && sd->intercepted_color_count < MAX_COLOR_IMAGES)
    {
        sd->intercepted_color[sd->intercepted_color_count++] = sc->stereo_images[0];
        STEREO_LOG(
            "[IMAGE TRACK ADD] stereo_images[0]=%p added to intercepted_color list "
            "(count=%u). Eden vkCmdBlitImage calls targeting this image will be "
            "intercepted by image_blit.c for double-layer copy.",
            (void *)(uintptr_t)sc->stereo_images[0],
            sd->intercepted_color_count);
    }

    VkImageViewCreateInfo vci = {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = sc->stereo_images[0],
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format   = sc->format,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2 },
    };
    STEREO_LOG_VERBOSE("CALL real CreateImageView");
    res = sd->real.CreateImageView(sd->real_device, &vci, NULL, &sc->stereo_views_arr[0]);
    STEREO_LOG_VERBOSE("RETURN real CreateImageView result=%d", res);
    if (res != VK_SUCCESS) return res;

    if (sd->upgraded_view_count < MAX_UPGRADED_VIEWS)
    {
        CHECK_ARRAY_COUNT(sd->upgraded_view_count, MAX_UPGRADED_VIEWS, "upgraded_view_count");
        sd->upgraded_views[sd->upgraded_view_count++] =
            sc->stereo_views_arr[0];
    
        //STEREO_LOG_VERBOSE(
        //    "[VIEW TRACK ADD NV3D] view=%p count=%u",
        //    sc->stereo_views_arr[0],
        //    sd->upgraded_view_count);
    }
    //STEREO_LOG_VERBOSE(
    //    "[NV3D TEST] alloc_alt_stereo_swapchain image=%p view=%p count=%u",
    //    sc->stereo_images[0],
    //    sc->stereo_views_arr[0],
    //    sc->image_count);

    /* CPU staging NOT created here — caller adds it for DX9, not for GPU compose */
    return VK_SUCCESS;
}

/* ── image untracking helper ─ */
static void remove_tracked_image(
    VkImage *arr,
    uint32_t *count,
    VkImage image)
{
    //STEREO_LOG_VERBOSE(
    //    "[IMAGE REMOVE SEARCH] image=%p count=%u first=%p",
    //    image,
    //    *count,
    //    *count ? arr[0] : VK_NULL_HANDLE);
    //STEREO_LOG_VERBOSE(
    //    "[IMAGE REMOVE SEARCH] image=%p count=%u",
    //    image,
    //    *count);
    //STEREO_LOG_VERBOSE(
    //    "[IMAGE TRACK SEARCH] image=%p count=%u",
    //    image,
    //    *count);
    for (uint32_t i = 0; i < *count; i++)
    {
        if (arr[i] == image)
        {
            //STEREO_LOG_VERBOSE(
            //    "[IMAGE REMOVE FOUND] image=%p slot=%u",
            //    image,
            //    i);
            //STEREO_LOG_VERBOSE(
            //    "[IMAGE REMOVE VALUE] image=%p last=%p count=%u",
            //    image,
            //    arr[*count - 1],
            //    *count);
            uint32_t last = --(*count);
            arr[i] = arr[last];

            //STEREO_LOG_VERBOSE(
            //    "[IMAGE TRACK REMOVE] image=%p slot=%u count=%u",
            //    image,
            //    i,
            //    *count);

            return;
        }
    }
    //STEREO_LOG_VERBOSE(
    //    "[IMAGE TRACK MISS] image=%p count=%u",
    //    image,
    //    *count);
    //STEREO_LOG_VERBOSE(
    //    "[IMAGE REMOVE MISS] image=%p count=%u",
    //    image,
    //    *count);
}

static void tracked_destroy_image(
    StereoDevice *sd,
    VkImage image,
    const char *site)
{
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY IMAGE] site=%s image=%p depth_count=%u color_count=%u",
    //    site,
    //    image,
    //    sd->intercepted_depth_count,
    //    sd->intercepted_color_count);

    remove_tracked_image(
        sd->intercepted_depth,
        &sd->intercepted_depth_count,
        image);

    remove_tracked_image(
        sd->intercepted_color,
        &sd->intercepted_color_count,
        image);

    sd->real.DestroyImage(
        sd->real_device,
        image,
        NULL);
}

/* ── vkCreateSwapchainKHR ──────────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateSwapchainKHR(VkDevice device,
                          const VkSwapchainCreateInfoKHR *pCreateInfo,
                          const VkAllocationCallbacks    *pAllocator,
                          VkSwapchainKHR                 *pSwapchain)
{
    STEREO_LOG("stereo_CreateSwapchainKHR: surface=%p extent=%ux%u layers=%u fmt=%u",
        (void*)(uintptr_t)pCreateInfo->surface,
        (unsigned)pCreateInfo->imageExtent.width,
        (unsigned)pCreateInfo->imageExtent.height,
        (unsigned)pCreateInfo->imageArrayLayers,
        (unsigned)pCreateInfo->imageFormat);
    STEREO_LOG_VERBOSE("CALLED stereo_CreateSwapchainKHR");
    STEREO_LOG_VERBOSE(
        "[CREATE SC] surface=%p old=%p",
        pCreateInfo->surface,
        pCreateInfo->oldSwapchain);
    StereoDevice *sd = stereo_device_from_handle(device);
    STEREO_LOG_VERBOSE(
        "SWAPCHAIN device=%p sd=%p real=%p",
        (void*)device,
        (void*)sd,
        sd ? (void*)sd->real_device : NULL);
    STEREO_LOG_VERBOSE(
        "[CREATE SC START] count=%u old=%p",
        sd->swapchain_count,
        pCreateInfo->oldSwapchain);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    if (!sd->stereo.enabled || sd->swapchain_count >= MAX_SWAPCHAINS)
        return sd->real.CreateSwapchainKHR(sd->real_device, pCreateInfo, pAllocator, pSwapchain);

    uint32_t app_w = pCreateInfo->imageExtent.width;
    uint32_t app_h = pCreateInfo->imageExtent.height;

    STEREO_LOG_VERBOSE(
        "[CREATE SC] swapchain_count=%u old=%p",
        sd->swapchain_count,
        pCreateInfo->oldSwapchain);

    StereoSwapchain *old_sc = NULL;

    if (pCreateInfo->oldSwapchain != VK_NULL_HANDLE)
    {
        old_sc =
            stereo_swapchain_lookup(
                sd,
                pCreateInfo->oldSwapchain);

        STEREO_LOG_VERBOSE(
            "[CREATE SC OLD LOOKUP] old=%p old_sc=%p",
            pCreateInfo->oldSwapchain,
            old_sc);
    }
    
    if (pCreateInfo->oldSwapchain != VK_NULL_HANDLE)
    {
        old_sc =
            stereo_swapchain_lookup(
                sd,
                pCreateInfo->oldSwapchain);
    
        STEREO_LOG_VERBOSE(
            "[CREATE SC OLD LOOKUP] old=%p old_sc=%p",
            pCreateInfo->oldSwapchain,
            old_sc);
    }
    
    StereoSwapchain *sc;
    
    if (old_sc)
    {
        sc = old_sc;
        sc->resize_reused = true;
    
        STEREO_LOG_VERBOSE(
            "[CREATE SC REUSE] sc=%p",
            sc);
    }
    else
    {
    sc = &sd->swapchains[sd->swapchain_count];

    memset(sc, 0, sizeof(*sc));
    sc->resize_reused = false;
    STEREO_LOG_VERBOSE(
        "[CREATE SC NEW] sc=%p count=%u reused=%d",
        sc,
        sd->swapchain_count,
        (int)sc->resize_reused);
    }

    sc->device     = sd->real_device;
    sc->app_width  = app_w;
    sc->app_height = app_h;
    sc->format     = pCreateInfo->imageFormat;
    sc->hwnd       = stereo_si_hwnd_for_surface(sd->si, pCreateInfo->surface);

    STEREO_LOG_VERBOSE(
        "CreateSwapchain: enabled=%d present_mode=%d",
        (int)sd->stereo.enabled,
        (int)sd->stereo.present_mode);

    StereoPresentMode req = sd->stereo.present_mode;

    STEREO_LOG_VERBOSE(
        "CreateSwapchain: req=%d stereo.enabled=%d",
        (int)req,
        (int)sd->stereo.enabled);

    if (req == STEREO_PRESENT_NV3DLIB)
    {
        STEREO_LOG_VERBOSE("[NV3D] requested");

        if (!nv3d_init(sd, app_w, app_h))
        {
            STEREO_ERR("[NV3D] init failed");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        if (sc->stereo_images)
        {
            STEREO_LOG_VERBOSE(
                "[NV3D] stereo_image[0]=%p",
                sc->stereo_images[0]);
        }
        STEREO_LOG_VERBOSE("[NV3D] init succeeded");

        VkResult nvres =
            alloc_alt_stereo_swapchain(sd, sc);

        if (nvres == VK_SUCCESS)
        {
            if (!setup_barrier_resources(sd, sc))
            {
                STEREO_ERR(
                    "[NV3D] setup_barrier_resources failed");

                nvres = VK_ERROR_INITIALIZATION_FAILED;
            }
        }

        STEREO_LOG_VERBOSE(
            "[NV3D] alloc_alt_stereo_swapchain=%d image_count=%u images=%p",
            nvres,
            sc->image_count,
            sc->stereo_images);

        STEREO_LOG_VERBOSE(
            "[NV3D] after alloc cmds=%p fences=%p",
            sc->barrier_cmds,
            sc->barrier_fences);

        if (nvres != VK_SUCCESS)
        {
            STEREO_ERR(
                "[NV3D] alloc_alt_stereo_swapchain failed");
            goto passthrough;
        }
        sc->present_mode  = STEREO_PRESENT_NV3DLIB;
        sc->stereo_active = true;
        sc->real_swapchain = VK_NULL_HANDLE;
        *pSwapchain = (VkSwapchainKHR)(uintptr_t)sc;
        sc->app_handle = *pSwapchain;
        STEREO_LOG_VERBOSE(
            "[CREATE SC] sc=%p app_handle=%p returned=%p",
            sc,
            sc->app_handle,
            *pSwapchain);
        CHECK_ARRAY_COUNT(sd->swapchain_count, MAX_SWAPCHAINS, "swapchain_count");
        if (pCreateInfo->oldSwapchain == VK_NULL_HANDLE)
            sd->swapchain_count++;

        STEREO_LOG_VERBOSE(
            "[NV3D] RETURNING NV3D SWAPCHAIN handle=%p",
            (void*)*pSwapchain);

        STEREO_LOG_VERBOSE(
            "[NV3D] return cmds=%p fences=%p",
            sc->barrier_cmds,
            sc->barrier_fences);

        if (sc->barrier_cmds)
        {
            STEREO_LOG_VERBOSE(
                "[NV3D] cmd0=%p",
                sc->barrier_cmds[0]);
        }

        if (sc->barrier_fences)
        {
            STEREO_LOG_VERBOSE(
                "[NV3D] fence0=%p",
                sc->barrier_fences[0]);
        }
        return VK_SUCCESS;
    }

    STEREO_LOG_VERBOSE(
        "CreateSwapchain: req=%d stereo.enabled=%d",
        (int)req,
        (int)sd->stereo.enabled);

    /* ── DXGI 1.2 + external memory ─────────────────────────────────── */
    if (req == STEREO_PRESENT_AUTO || req == STEREO_PRESENT_DXGI) {
        bool dxgi_ok = false;
        HANDLE nt_handle = NULL;
        if (sd->dxgi_init_in_progress) goto passthrough;
        if (sc->hwnd && dxgi_device_init(sd)) {
            dxgi_stereo_activate(sd);
            sd->dxgi_init_in_progress = true;
            if (dxgi_sc_create(sd, sc, &nt_handle)) {
                dxgi_ok = true;
                sd->dxgi_init_in_progress = false;
            } else {
                sd->dxgi_init_in_progress = false;
                dxgi_sc_destroy(sc);
            }
        }
        if (dxgi_ok) {
            sc->image_count      = 1;
            sc->stereo_images    = calloc(1, sizeof(VkImage));
            sc->stereo_memory    = calloc(1, sizeof(VkDeviceMemory));
            sc->stereo_views_arr = calloc(1, sizeof(VkImageView));
            sc->barrier_cmds     = calloc(1, sizeof(VkCommandBuffer));
            sc->barrier_fences   = calloc(1, sizeof(VkFence));
            if (!sc->stereo_images || !sc->stereo_memory || !sc->stereo_views_arr ||
                !sc->barrier_cmds  || !sc->barrier_fences) {
                dxgi_sc_destroy(sc);
                if (req == STEREO_PRESENT_DXGI) goto passthrough;
                goto try_dx9;
            }
            VkResult res = alloc_external_stereo_image(sd, sc,
                &sc->stereo_images[0], &sc->stereo_memory[0]);
            if (res != VK_SUCCESS) {
                dxgi_sc_destroy(sc);
                if (req == STEREO_PRESENT_DXGI) goto passthrough;
                goto try_dx9;
            }
            VkImageViewCreateInfo vci = {
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = sc->stereo_images[0],
                .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                .format   = sc->format,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2 },
            };
            STEREO_LOG_VERBOSE("CALL real CreateImageView");
            res = sd->real.CreateImageView(sd->real_device, &vci, NULL, &sc->stereo_views_arr[0]);
            STEREO_LOG_VERBOSE("RETURN real CreateImageView result=%d", res);
            if (res != VK_SUCCESS) { dxgi_sc_destroy(sc); goto try_dx9; }
            if (!setup_barrier_resources(sd, sc)) { dxgi_sc_destroy(sc); goto try_dx9; }
            sc->present_mode  = STEREO_PRESENT_DXGI;
            sc->dxgi_mode     = true;
            sc->stereo_active = true;
            sc->real_swapchain = VK_NULL_HANDLE;
            *pSwapchain       = (VkSwapchainKHR)(uintptr_t)sc;
            sc->app_handle    = *pSwapchain;
            STEREO_LOG_VERBOSE(
                "[CREATE SC] sc=%p app_handle=%p returned=%p",
                sc,
                sc->app_handle,
                *pSwapchain);
            sd->stereo_w = app_w; sd->stereo_h = app_h;
            CHECK_ARRAY_COUNT(sd->swapchain_count, MAX_SWAPCHAINS, "swapchain_count");
            if (pCreateInfo->oldSwapchain == VK_NULL_HANDLE)
                sd->swapchain_count++;
            STEREO_LOG_VERBOSE("DXGI stereo swapchain (external mem): %ux%u  handle=%p",
                       app_w, app_h, (void*)*pSwapchain);
            return VK_SUCCESS;
        }
        if (req == STEREO_PRESENT_DXGI) { STEREO_ERR("DXGI forced but failed"); goto passthrough; }
    }

try_dx9:
    if (req == STEREO_PRESENT_AUTO || req == STEREO_PRESENT_DX9) {
        if (!sd->d3d11_ok) dxgi_device_init(sd);
        if (sc->hwnd && dx9_init(sd, sc)) {
            VkResult res = alloc_alt_stereo_swapchain(sd, sc);
            if (res == VK_SUCCESS) res = alt_cpu_staging_init(sd, sc); /* DX9 needs CPU staging */
            if (res == VK_SUCCESS && setup_barrier_resources(sd, sc)) {
                sc->present_mode  = STEREO_PRESENT_DX9;
                sc->dxgi_mode     = false;
                sc->stereo_active = true;
                sc->real_swapchain = VK_NULL_HANDLE;
                *pSwapchain       = (VkSwapchainKHR)(uintptr_t)sc;
                sc->app_handle    = *pSwapchain;
                STEREO_LOG_VERBOSE(
                    "[CREATE SC] sc=%p app_handle=%p returned=%p",
                    sc,
                    sc->app_handle,
                    *pSwapchain);
                sd->stereo_w = app_w; sd->stereo_h = app_h;
                CHECK_ARRAY_COUNT(sd->swapchain_count, MAX_SWAPCHAINS, "swapchain_count");
                if (pCreateInfo->oldSwapchain == VK_NULL_HANDLE)
                    sd->swapchain_count++;
                STEREO_LOG_VERBOSE("DX9 stereo swapchain: %ux%u  handle=%p", app_w, app_h, (void*)*pSwapchain);
                return VK_SUCCESS;
            }
        }
        if (req == STEREO_PRESENT_DX9) { STEREO_ERR("DX9 forced but failed"); goto passthrough; }
        req = STEREO_PRESENT_SBS;
    }

    /* ── GPU blit compose (SBS / TAB / Interlaced) ───────────────────── */
    if (req == STEREO_PRESENT_SBS  ||
        req == STEREO_PRESENT_TAB  ||
        req == STEREO_PRESENT_INTERLACED) {
            STEREO_LOG("[CREATE SC SBS] try w=%u h=%u req_mode=%d hwnd=%p surface=%p enabled=%d",
                   app_w, app_h, (int)req,
                   (void*)(uintptr_t)sc->hwnd,
                   (void*)(uintptr_t)pCreateInfo->surface,
                   (int)sd->stereo.enabled);
        if (!sc->hwnd) {
            STEREO_LOG("[CREATE SC SBS] SKIP hwnd=NULL (no window surface — not the main presentation window). w=%u h=%u",
                       app_w, app_h);
        }
        if (sc->hwnd && gpu_compose_sc_init(sd, sc, pCreateInfo->surface)) {
            VkResult res = alloc_alt_stereo_swapchain(sd, sc);
            if (res == VK_SUCCESS && setup_barrier_resources(sd, sc)) {
                sc->present_mode  = req;
                sc->dxgi_mode     = false;
                sc->stereo_active = true;
                *pSwapchain       = (VkSwapchainKHR)(uintptr_t)sc;
                sc->app_handle    = *pSwapchain;
                STEREO_LOG(
                    "SWAPCHAIN_SBS_ACTIVE sc=%p app=%p real=%p active=%d mode=%d %ux%u",
                    sc,
                    (void*)sc->app_handle,
                    (void*)sc->real_swapchain,
                    (int)sc->stereo_active,
                    (int)req,
                    app_w,
                    app_h);
                STEREO_LOG_VERBOSE(
                    "[CREATE SC GPU FINAL] sc=%p app=%p real=%p active=%d count=%u",
                    sc,
                    sc->app_handle,
                    sc->real_swapchain,
                    (int)sc->stereo_active,
                    sd->swapchain_count);
                CHECK_ARRAY_COUNT(sd->swapchain_count, MAX_SWAPCHAINS, "swapchain_count");
                if (!old_sc)
                    sd->swapchain_count++;
                STEREO_LOG_VERBOSE(
                    "[CREATE SC GPU FINAL] sc=%p app=%p real=%p active=%d count=%u",
                    sc,
                    sc->app_handle,
                    sc->real_swapchain,
                    (int)sc->stereo_active,
                    sd->swapchain_count);
                sd->stereo_w = app_w;
                sd->stereo_h = app_h;
                /* Always initialise CPU readback staging for side-by-side
                 * regardless of image_shift:
                 *   image_shift=1 : required by compose_present pixel-level shift
                 *   image_shift=0 : single diagnostic frame compares layer 0 vs
                 *                   layer 1 pixels to prove whether gl_ViewIndex
                 *                   is populated by the driver. */
                if (req == STEREO_PRESENT_SBS) {
                    bool ok = compose_init(sd, sc) &&
                              (alt_cpu_staging_init(sd, sc) == VK_SUCCESS);
                    if (!ok) {
                        STEREO_LOG("[CREATE SC] CPU staging init FAILED "
                                   "(comp_ok=%d cpu_ok=%d). "
                                   "Layer diagnostic + image_shift unavailable.",
                                   (int)sd->comp_ok, (int)sc->cpu_ok);
                    } else {
                        STEREO_LOG("[CREATE SC] CPU staging READY comp_ok=%d cpu_ok=%d "
                                   "(for layer diagnostic + image_shift fallback).",
                                   (int)sd->comp_ok, (int)sc->cpu_ok);
                    }
                }
                STEREO_LOG_VERBOSE(
                    "[CREATE SC GPU] sc=%p app_handle=%p returned=%p",
                    sc,
                    sc->app_handle,
                    *pSwapchain);
                STEREO_LOG_VERBOSE(
                    "GPU-blit stereo swapchain (mode=%d): %ux%u  handle=%p",
                    (int)req,
                    app_w,
                    app_h,
                    (void*)*pSwapchain);
                return VK_SUCCESS;
            }
            /* GPU compose init failed — fall to passthrough */
            //STEREO_LOG_VERBOSE("[DESTROY SC] before gpu_compose_sc_destroy");
            gpu_compose_sc_destroy(sd, sc);
            //STEREO_LOG_VERBOSE("[DESTROY SC] after gpu_compose_sc_destroy");
            if (sc->real_swapchain) {
                //STEREO_LOG_VERBOSE(
                //    "[COMPOSE DESTROY] (swapchain.c) destroying=%p",
                //    sc->real_swapchain);
                //STEREO_LOG_VERBOSE(
                //    "[COMPOSE DESTROY] sc=%p app=%p real=%p",
                //    sc,
                //    sc->app_handle,
                //    sc->real_swapchain);
                sd->real.DestroySwapchainKHR(sd->real_device, sc->real_swapchain, NULL);
                //STEREO_LOG_VERBOSE(
                //    "[COMPOSE DESTROY] (swapchain.c) destroyed=%p",
                //    sc->real_swapchain);
                sc->real_swapchain = VK_NULL_HANDLE;
            }
        }
    }

passthrough:
    STEREO_LOG("[CREATE SC PASSTHROUGH] w=%u h=%u req_mode=%d hwnd=%p stereo_active=0 "
               "(All stereo compose paths failed or were skipped — forwarding real swapchain).",
               app_w, app_h, (int)req, (void*)(uintptr_t)sc->hwnd);
    STEREO_LOG_VERBOSE(
        "[PASSTHROUGH] entering real CreateSwapchainKHR old=%p",
        pCreateInfo->oldSwapchain);
    sc->stereo_active = false;
    STEREO_LOG_VERBOSE(
        "[PASSTHROUGH] calling real CreateSwapchainKHR");

    VkSwapchainCreateInfoKHR ci = *pCreateInfo;

    if (pCreateInfo->oldSwapchain != VK_NULL_HANDLE)
    {
        StereoSwapchain *old_sc =
            stereo_swapchain_lookup(sd, pCreateInfo->oldSwapchain);
        STEREO_LOG_VERBOSE(
            "[PASSTHROUGH LOOKUP] old=%p old_sc=%p real=%p",
            pCreateInfo->oldSwapchain,
            old_sc,
            old_sc ? old_sc->real_swapchain : VK_NULL_HANDLE);

        if (old_sc)
        {
            ci.oldSwapchain = old_sc->real_swapchain;

            STEREO_LOG_VERBOSE(
                "[PASSTHROUGH] translated old swapchain %p -> %p",
                pCreateInfo->oldSwapchain,
                ci.oldSwapchain);
        }
        else
        {
            ci.oldSwapchain = pCreateInfo->oldSwapchain;

            STEREO_LOG_VERBOSE(
                "[PASSTHROUGH] forwarding unknown old swapchain %p",
                ci.oldSwapchain);
        }

    }
    STEREO_LOG_VERBOSE("CALL real CreateSwapchainKHR");
    VkResult res =
        sd->real.CreateSwapchainKHR(
            sd->real_device,
            &ci,
            pAllocator,
            pSwapchain);
    STEREO_LOG_VERBOSE("RETURN real CreateSwapchainKHR result=%d", res);
    STEREO_LOG_VERBOSE(
        "[PASSTHROUGH] returned %d swapchain=%p",
        (int)res,
        res == VK_SUCCESS ? *pSwapchain : VK_NULL_HANDLE);
    if (res == VK_SUCCESS) {
        sc->real_swapchain = *pSwapchain;
        sc->app_handle     = *pSwapchain;
        sc->stereo_active  = false;
        CHECK_ARRAY_COUNT(sd->swapchain_count, MAX_SWAPCHAINS, "swapchain_count");
        if (pCreateInfo->oldSwapchain == VK_NULL_HANDLE)
            sd->swapchain_count++;
    }
    return res;
}

/* ── vkDestroySwapchainKHR ──────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain,
                            const VkAllocationCallbacks *pAllocator)
{
    STEREO_LOG_VERBOSE("CALLED stereo_DestroySwapchainKHR");
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC ENTRY] swapchain=%p",
    //    swapchain);
    StereoDevice *sd = stereo_device_from_handle(device);
    if (!sd) return;
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC START] count=%u",
    //    sd->swapchain_count);

    StereoSwapchain *sc = stereo_swapchain_lookup(sd, swapchain);

    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC LOOKUP RESULT] app=%p sc=%p active=%d images=%u",
    //    swapchain,
    //    sc,
    //    sc ? (int)sc->stereo_active : -1,
    //    sc ? sc->image_count : 0);
    if (sc && sc->resize_reused)
    {
        //STEREO_LOG_VERBOSE(
        //    "[DESTROY SC] ignoring recycled resize swapchain app=%p sc=%p",
        //    swapchain,
        //    sc);
    
        sc->resize_reused = false;
        return;
    }

    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC] present_mode=%d active=%d app=%p real=%p",
    //    sc ? (int)sc->present_mode : -1,
    //    sc ? (int)sc->stereo_active : -1,
    //    sc ? sc->app_handle : VK_NULL_HANDLE,
    //    sc ? sc->real_swapchain : VK_NULL_HANDLE);
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC LOOKUP] app=%p sc=%p",
    //    swapchain,
    //    sc);

    if (sc) {
        //STEREO_LOG_VERBOSE(
        //    "[DESTROY SC] stereo_active=%d",
        //    sc ? (int)sc->stereo_active : -1);

        //STEREO_LOG_VERBOSE(
        //    "[DESTROY SC] image_count=%u stereo_images=%p stereo_views=%p",
        //    sc->image_count,
        //    sc->stereo_images,
        //    sc->stereo_views_arr);

        for (uint32_t i = 0; i < sc->image_count; i++)
        {
            //STEREO_LOG_VERBOSE("[DESTROY SC] image %u", i);

            if (sc->stereo_views_arr && sc->stereo_views_arr[i])
            {
                //STEREO_LOG_VERBOSE(
                //    "[DESTROY SC] destroy imageview %u view=%p",
                //    i,
                //    sc->stereo_views_arr[i]);
                stereo_DestroyImageView(
                    device,
                    sc->stereo_views_arr[i],
                    NULL);
            }

            if (sc->stereo_images && sc->stereo_images[i])
            {
                bool depth_match = false;
                bool color_match = false;
                
                for (uint32_t d = 0; d < sd->intercepted_depth_count; d++)
                {
                    if (sd->intercepted_depth[d] == sc->stereo_images[i])
                    {
                        depth_match = true;
                        break;
                    }
                }
                
                for (uint32_t c = 0; c < sd->intercepted_color_count; c++)
                {
                    if (sd->intercepted_color[c] == sc->stereo_images[i])
                    {
                        color_match = true;
                        break;
                    }
                }
                //STEREO_LOG_VERBOSE(
                //    "[TRACK REMOVE ATTEMPT] image=%p depth_match=%d color_match=%d",
                //    sc->stereo_images[i],
                //    depth_match,
                //    color_match);
                //STEREO_LOG_VERBOSE(
                //    "[IMAGE REMOVE TRY] image=%p sc=%p",
                //    sc->stereo_images[i],
                //    sc);
                remove_tracked_image(
                    sd->intercepted_depth,
                    &sd->intercepted_depth_count,
                    sc->stereo_images[i]);
                //STEREO_LOG_VERBOSE(
                //    "[IMAGE REMOVE TRY] image=%p sc=%p",
                //    sc->stereo_images[i],
                //    sc);
                remove_tracked_image(
                    sd->intercepted_color,
                    &sd->intercepted_color_count,
                    sc->stereo_images[i]);
                tracked_destroy_image(
                    sd,
                    sc->stereo_images[i],
                    "swapchain stereo image");
            }

            if (sc->stereo_memory && sc->stereo_memory[i])
            {
                //STEREO_LOG_VERBOSE("[DESTROY SC] free memory %u", i);
                sd->real.FreeMemory(
                    sd->real_device,
                    sc->stereo_memory[i],
                    NULL);
            }
            if (sc->barrier_fences && sc->barrier_fences[i])
                sd->real.DestroyFence(sd->real_device, sc->barrier_fences[i], NULL);
        }
        free(sc->stereo_views_arr);
        free(sc->stereo_images);
        //STEREO_LOG_VERBOSE(
        //    "[IMAGE TRACK COUNTS] depth=%u color=%u",
        //    sd->intercepted_depth_count,
        //    sd->intercepted_color_count);
        free(sc->stereo_memory);
        free(sc->barrier_cmds);
        free(sc->barrier_fences);

        sc->stereo_views_arr = NULL;
        sc->stereo_images    = NULL;
        sc->stereo_memory    = NULL;
        sc->barrier_cmds     = NULL;
        sc->barrier_fences   = NULL;
        sc->image_count      = 0;

        if (sc->barrier_pool)
            sd->real.DestroyCommandPool(sd->real_device, sc->barrier_pool, NULL);
        sc->barrier_pool = VK_NULL_HANDLE;
        STEREO_LOG_VERBOSE(
            "[NV3D] QueuePresent mode=%d sc=%p",
            (int)sc->present_mode,
            sc);
        if (sc->present_mode == STEREO_PRESENT_NV3DLIB)
            nv3d_destroy(sd);

        //STEREO_LOG_VERBOSE("[DESTROY SC] before gpu_compose_sc_destroy");
        gpu_compose_sc_destroy(sd, sc);     /* semaphores + comp_sc_images array */
        //STEREO_LOG_VERBOSE("[DESTROY SC] after gpu_compose_sc_destroy");
        //STEREO_LOG_VERBOSE("[DESTROY SC] before alt_cpu_staging_destroy");
        alt_cpu_staging_destroy(sd, sc);    /* DX9 CPU staging (no-op if unused) */
        //STEREO_LOG_VERBOSE("[DESTROY SC] after alt_cpu_staging_destroy");
        //STEREO_LOG_VERBOSE("[DESTROY SC] before dxgi_sc_destroy");
        dxgi_sc_destroy(sc);
        //STEREO_LOG_VERBOSE("[DESTROY SC] after dxgi_sc_destroy");

        /* real_swapchain: GPU compose output SC or passthrough SC */
        if (sc->real_swapchain)
        {
            //STEREO_LOG_VERBOSE(
            //    "[DESTROY SC] app=%p sc=%p real=%p",
            //    swapchain,
            //    sc,
            //    sc->real_swapchain);
            //STEREO_LOG_VERBOSE(
            //    "[COMPOSE DESTROY] (swapchain.c) destroying=%p",
            //    sc->real_swapchain);
            sd->real.DestroySwapchainKHR(
                sd->real_device,
                sc->real_swapchain,
                pAllocator);
            //STEREO_LOG_VERBOSE(
            //    "[COMPOSE DESTROY] (swapchain.c) destroyed=%p",
            //    sc->real_swapchain);
            sc->real_swapchain = VK_NULL_HANDLE;
        }

        //STEREO_LOG_VERBOSE(
        //    "[DESTROY SC] keeping slot alive sc=%p",
        //    sc);

        /* leave structure in table */
        sc->stereo_active = false;

    } else {
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC PASSTHROUGH] BEFORE destroy swapchain=%p",
    //    swapchain);

    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC PASSTHROUGH] device=%p",
    //    sd->real_device);

    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC PASSTHROUGH] calling real destroy device=%p swapchain=%p",
    //    sd->real_device,
    //    swapchain);

    sd->real.DestroySwapchainKHR(
        sd->real_device,
        swapchain,
        pAllocator);

    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC PASSTHROUGH] real destroy returned");

    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC PASSTHROUGH] AFTER destroy swapchain=%p",
    //    swapchain);
    }
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY SC END] count=%u",
    //    sd->swapchain_count);
}

/* ── vkGetSwapchainImagesKHR ────────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_GetSwapchainImagesKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    uint32_t *pCount,
    VkImage *pImages)
{
    STEREO_LOG_VERBOSE("CALLED stereo_GetSwapchainImagesKHR");
    STEREO_LOG_VERBOSE(
        "GetSwapchainImagesKHR swapchain=%p count_ptr=%p images_ptr=%p",
        swapchain,
        pCount,
        pImages);
    StereoDevice *sd = stereo_device_from_handle(device);
    if (!sd) return VK_ERROR_DEVICE_LOST;

    StereoSwapchain *sc = stereo_swapchain_lookup(sd, swapchain);
    //STEREO_LOG_VERBOSE(
    //    "[GET IMAGES] sc=%p",
    //    sc);

    //STEREO_LOG_VERBOSE(
    //    "[GET IMAGES] stereo_active addr=%p",
    //    &sc->stereo_active);
    //STEREO_LOG_VERBOSE(
    //    "[GET IMAGES LOOKUP] app=%p sc=%p real=%p active=%d",
    //    swapchain,
    //    sc,
    //    sc ? sc->real_swapchain : VK_NULL_HANDLE,
    //    sc ? sc->stereo_active : -1);
    if (!sc || !sc->stereo_active)
    {
        //STEREO_LOG_VERBOSE(
        //    "[GET IMAGES PASSTHROUGH] sc=%p active=%d real=%p",
        //    sc,
        //    sc ? (int)sc->stereo_active : -1,
        //    sc ? sc->real_swapchain : VK_NULL_HANDLE);

        if (sc && sc->real_swapchain == VK_NULL_HANDLE)
        {
            //STEREO_ERR(
            //    "[GET IMAGES] called on destroyed stereo swapchain");
            return VK_ERROR_OUT_OF_DATE_KHR;
        }

        VkSwapchainKHR real =
            sc ? sc->real_swapchain : swapchain;

        return sd->real.GetSwapchainImagesKHR(
            sd->real_device,
            real,
            pCount,
            pImages);
    }
    STEREO_LOG_VERBOSE(
        "GetSwapchainImagesKHR stereo=%d image_count=%u",
        sc ? sc->stereo_active : 0,
        sc ? sc->image_count : 0);
    if (!pImages)
    {
        STEREO_LOG_VERBOSE(
            "[NV3D TEST] count query image_count=%u",
            sc->image_count);

        *pCount = sc->image_count;
        return VK_SUCCESS;
    }
    uint32_t copy = (*pCount < sc->image_count) ? *pCount : sc->image_count;
    STEREO_LOG_VERBOSE(
        "GetSwapchainImagesKHR returning %u images stereo_images=%p",
        copy,
        sc->stereo_images);
    for (uint32_t i = 0; i < copy; i++)
    {
        pImages[i] = sc->stereo_images[i];

        STEREO_LOG_VERBOSE(
            "[NV3D TEST] image[%u]=%p",
            i,
            (void*)pImages[i]);
    }
    *pCount = copy;
    return (copy < sc->image_count) ? VK_INCOMPLETE : VK_SUCCESS;

}

/* ── vkAcquireNextImageKHR ───────────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain,
                            uint64_t timeout, VkSemaphore semaphore,
                            VkFence fence, uint32_t *pImageIndex)
{
    STEREO_LOG_VERBOSE("CALLED stereo_AcquireNextImageKHR");
    StereoDevice *sd = stereo_device_from_handle(device);
    //STEREO_LOG_VERBOSE(
    //    "[NV3D] acquire gfx_queue=%p",
    //    sd ? sd->gfx_queue : NULL);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    //STEREO_LOG_VERBOSE("stereo_AcquireNextImageKHR: sc=%p", (void*)swapchain);

    StereoSwapchain *sc = stereo_swapchain_lookup(sd, swapchain);

    //STEREO_LOG_VERBOSE(
    //    "[ACQUIRE LOOKUP] app=%p sc=%p real=%p active=%d",
    //    swapchain,
    //    sc,
    //    sc ? sc->real_swapchain : VK_NULL_HANDLE,
    //    sc ? sc->stereo_active : -1);
    //STEREO_LOG_VERBOSE(
    //    "stereo_AcquireNextImageKHR: sc=%p mode=%d real_sc=%p",
    //    sc,
    //    sc ? (int)sc->present_mode : -1,
    //    sc ? (void*)sc->real_swapchain : 0);

    if (sc &&
        sc->present_mode == STEREO_PRESENT_NV3DLIB)
    {
        STEREO_LOG_VERBOSE("[NV3D] AcquireNextImageKHR begin");

    if (semaphore != VK_NULL_HANDLE)
    {
        VkSubmitInfo sig = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = 1, 
            .pSignalSemaphores = &semaphore,
        };

        STEREO_LOG_VERBOSE(
            "[NV3D] signaling acquire semaphore %p",
            semaphore);

        sd->real.QueueSubmit(
            sd->gfx_queue,
            1,
            &sig,
            VK_NULL_HANDLE);
    }

    if (fence != VK_NULL_HANDLE)
    {
        sd->real.ResetFences(
            sd->real_device,
            1,
            &fence);

        sd->real.QueueSubmit(
            sd->gfx_queue,
            0,
            NULL,
            fence);

        STEREO_LOG_VERBOSE(
            "[NV3D] signaling acquire fence %p",
            fence);
    }

    if (pImageIndex)
        *pImageIndex = 0;

    STEREO_LOG_VERBOSE(
        "[NV3D] AcquireNextImageKHR return success");

    return VK_SUCCESS;
    }

    if (!sc || !sc->stereo_active)
    {
        STEREO_LOG_VERBOSE(
            "[ACQUIRE PASSTHROUGH] sc=%p active=%d real=%p",
            sc,
            sc ? (int)sc->stereo_active : -1,
            sc ? sc->real_swapchain : VK_NULL_HANDLE);

        if (sc && sc->real_swapchain == VK_NULL_HANDLE)
        {
            STEREO_ERR(
                "[ACQUIRE] called on destroyed stereo swapchain");
            return VK_ERROR_OUT_OF_DATE_KHR;
        }

        VkSwapchainKHR real =
            sc ? sc->real_swapchain : swapchain;

        return sd->real.AcquireNextImageKHR(
            sd->real_device,
            real,
            timeout,
            semaphore,
            fence,
            pImageIndex);
    }

    /* Wait for the previous frame's GPU work (DXGI barrier or GPU blit) to
     * complete before allowing the app to render into stereo_images[0] again.
     * barrier_fences[0] starts SIGNALED so the very first acquire never blocks. */
    if (sc->barrier_fences && sc->barrier_fences[0]) {
        VkResult wres = sd->real.WaitForFences(
            sd->real_device, 1, &sc->barrier_fences[0], VK_TRUE, timeout);
        if (wres != VK_SUCCESS) return wres;
    }

    /* Signal app semaphore/fence so it knows the image is available */
    if (semaphore != VK_NULL_HANDLE || fence != VK_NULL_HANDLE) {
        VkSubmitInfo sig = {
            .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = (semaphore != VK_NULL_HANDLE) ? 1 : 0,
            .pSignalSemaphores    = &semaphore,
        };
        if (sd->gfx_queue) sd->real.QueueSubmit(sd->gfx_queue, 1, &sig, fence);
    }
    *pImageIndex = 0;
    return VK_SUCCESS;
}

/* ── vkQueuePresentKHR ───────────────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo)
{
    /* Raw entry trace — first 8 calls, UNCONDITIONAL, no dedup, before ANY
     * swapchain lookup.  This catches the case where the 1280×720 main
     * window's swapchain handle is simply unknown to our ICD (was created
     * through a path we didn't hook, e.g. headless / external swapchain). */
    {
        static volatile LONG s_entry_count = 0;
        LONG n = InterlockedIncrement(&s_entry_count);
        if (n <= 8) {
            uint32_t scount = pPresentInfo ? pPresentInfo->swapchainCount : 0;
            char handles[256]; handles[0] = 0;
            size_t off = 0;
            for (uint32_t p = 0; p < scount && off < sizeof(handles)-32; p++) {
                int wr = _snprintf(handles + off, sizeof(handles) - off,
                                   " [%u]=%p", p,
                                   pPresentInfo ? (void*)(uintptr_t)pPresentInfo->pSwapchains[p] : NULL);
                if (wr > 0) off += (size_t)wr;
            }
            STEREO_LOG("[QPE#%ld] QueuePresentKHR scount=%u handles:%s",
                       (long)n, scount, handles);
        }
    }

    extern StereoDevice g_devices[];
    extern uint32_t     g_device_count;

    StereoDevice    *sd = NULL;
    StereoSwapchain *sc = NULL;
    for (uint32_t d = 0; d < g_device_count && !sd; d++) {
        for (uint32_t p = 0; p < pPresentInfo->swapchainCount; p++) {
            StereoSwapchain *found = stereo_swapchain_lookup(
                &g_devices[d], pPresentInfo->pSwapchains[p]);
            if (found) { sd = &g_devices[d]; sc = found; break; }
        }
    }

    if (!sd || !sc || !sd->stereo.enabled || !sc->stereo_active) {
        StereoDevice *fwd = sd ? sd : (g_device_count > 0 ? &g_devices[0] : NULL);
        {
            /* Use the raw VkSwapchainKHR handle value as dedup key so we
             * reliably track the MAIN WINDOW swapchain even when our
             * StereoSwapchain wrapper lookup fails (sc==NULL). */
            static uint64_t s_pass_trace_key[32] = {0};
            static int      s_pass_trace_next = 0;
            uint64_t key;
            if (sc) {
                key = (uint64_t)(uintptr_t)sc;
            } else if (pPresentInfo && pPresentInfo->swapchainCount > 0) {
                key = (uint64_t)(uintptr_t)pPresentInfo->pSwapchains[0];
            } else {
                key = (uint64_t)(uintptr_t)pPresentInfo;
            }
            int found = 0;
            for (int t = 0; t < s_pass_trace_next && !found; t++)
                if (s_pass_trace_key[t] == key) found = 1;
            if (!found && s_pass_trace_next < 32) {
                s_pass_trace_key[s_pass_trace_next++] = key;
                uint32_t w = sc ? sc->app_width  : 0;
                uint32_t h = sc ? sc->app_height : 0;
                int pmode  = sc ? (int)sc->present_mode : -1;
                int active = sc ? (int)sc->stereo_active : 0;
                STEREO_LOG(
                    "[QPASS PASSTHROUGH] key=0x%llx sc=%p sd_valid=%d "
                    "stereo.enabled=%d stereo_active=%d present_mode=%d "
                    "w=%u h=%u hwnd=%p real_sc0=%p",
                    (unsigned long long)key,
                    (void*)sc,
                    (int)(sd != NULL),
                    sd ? (int)sd->stereo.enabled : 0,
                    active, pmode, w, h,
                    sc ? (void*)(uintptr_t)sc->hwnd : NULL,
                    (pPresentInfo && pPresentInfo->swapchainCount > 0)
                        ? (void*)(uintptr_t)pPresentInfo->pSwapchains[0] : NULL);
            }
        }
        if (!fwd) return VK_ERROR_DEVICE_LOST;
        /* Still poll hotkeys even in passthrough mode so user can adjust */
        if (sd) hotkeys_poll(sd);
        return fwd->real.QueuePresentKHR(queue, pPresentInfo);
    }

    hotkeys_poll(sd);

    VkResult result = VK_SUCCESS;
    for (uint32_t i = 0; i < pPresentInfo->swapchainCount; i++) {
        StereoSwapchain *sc_i = stereo_swapchain_lookup(sd, pPresentInfo->pSwapchains[i]);
        if (!sc_i || !sc_i->stereo_active) continue;

        uint32_t           wcount = (i == 0) ? pPresentInfo->waitSemaphoreCount : 0;
        const VkSemaphore *wsems  = (i == 0) ? pPresentInfo->pWaitSemaphores    : NULL;

        /* Universal per-swapchain trace (fires once per sc_i pointer) — runs
         * BEFORE the present_mode switch so we see EVERY active swapchain,
         * including ones that fall into DXGI/DX9/NV3D/passthrough.  This is
         * critical for figuring out why the 1280×720 main window never enters
         * the SBS compose path even though the game has rendered a frame. */
        {
            static uint64_t s_all_trace_key[32] = {0};
            static int      s_all_trace_next = 0;
            uint64_t key = (uint64_t)(uintptr_t)sc_i;
            int found = 0;
            for (int t = 0; t < s_all_trace_next && !found; t++)
                if (s_all_trace_key[t] == key) found = 1;
            if (!found && s_all_trace_next < 32) {
                s_all_trace_key[s_all_trace_next++] = key;
                int pm = (int)sc_i->present_mode;
                const char *pm_name =
                    (pm==0)?"AUTO":
                    (pm==1)?"DXGI":
                    (pm==2)?"DX9":
                    (pm==3)?"SBS":
                    (pm==4)?"TAB":
                    (pm==5)?"ILACED":
                    (pm==6)?"MONO":
                    (pm==7)?"NV3D":"?";
                STEREO_LOG(
                    "[QPRESS ALL] sc=%p mode=%s(%d) w=%u h=%u "
                    "stereo_active=%d cpu_ok=%d comp_ok=%d hwnd=%p",
                    (void*)sc_i, pm_name, pm,
                    sc_i->app_width, sc_i->app_height,
                    (int)sc_i->stereo_active,
                    (int)sc_i->cpu_ok,
                    (int)sd->comp_ok,
                    (void*)(uintptr_t)sc_i->hwnd);
            }
        }

        VkResult pr;
        switch (sc_i->present_mode) {
        case STEREO_PRESENT_DXGI:
            pr = stereo_dxgi_present(sd, queue, sc_i, 0, wcount, wsems);
            break;
        case STEREO_PRESENT_DX9:
            pr = dx9_present(sd, sc_i, queue, wcount, wsems);
            break;
        case STEREO_PRESENT_NV3DLIB:
            pr = nv3d_present(
                sd,
                sc_i,
                queue,
                wcount,
                wsems);
            break;
        case STEREO_PRESENT_SBS:
        case STEREO_PRESENT_TAB:
        case STEREO_PRESENT_INTERLACED: {
            /* Trace every entering frame's geometry + staging flags (once per
             * swapchain) so we can see whether the 1280×720 main frame ever
             * enters the SBS compose path at all, and whether cpu_ok is set on
             * it.  Without this log the one-shot diagnostic could silently
             * fail because the large swapchain never has cpu staging. */
            {
                static uint64_t s_trace_key[(32*2)] = {0};
                static int      s_trace_next = 0;
                uint64_t key = (uint64_t)(uintptr_t)sc_i;
                int found = 0;
                for (int i = 0; i < s_trace_next && !found; i++)
                    if (s_trace_key[i*2+0] == key) found = 1;
                if (!found && s_trace_next < 32) {
                    s_trace_key[s_trace_next*2+0] = key;
                    s_trace_next++;
                    StereoPresentMode pm = sc_i->present_mode;
                    STEREO_LOG(
                        "[PRESENT TRACE] sc=%p mode=%s w=%u h=%u stereo_active=%d "
                        "comp_ok=%d cpu_ok=%d comp=%d dxgi=%d",
                        (void*)sc_i,
                        (pm==STEREO_PRESENT_SBS)?"SBS":(pm==STEREO_PRESENT_TAB)?"TAB":(pm==STEREO_PRESENT_INTERLACED)?"ILAC":"?",
                        sc_i->app_width, sc_i->app_height,
                        (int)sc_i->stereo_active,
                        (int)sd->comp_ok, (int)sc_i->cpu_ok,
                        (int)(sc_i->real_swapchain != VK_NULL_HANDLE),
                        (int)sc_i->dxgi_mode);
                }
            }
            /* First-present diagnostic (one-shot with RE-ARM):
             *   alt_cpu_readback + stereo_diagnose_layer_compare.
             * This is a pure read — no GDI BitBlt, no display, no switch of
             * swapchain owner.  It tells us whether the driver populated
             * gl_ViewIndex per view (differing>0) or not (differing=0).
             *
             * SOFT one-shot: stereo_diagnose_layer_compare returns BLACK
             * when the captured frame is 100% clear-only / pure black
             * (e.g. first frame after a swapchain resize / window rebuild).
             * In that case we RE-ARM BOTH gates so the next real 3D scene
             * frame is compared instead of locking to a useless verdict.
             * After a real-content verdict (COMPLETED) both gates lock to 1
             * and we never retry.                                      */
            static int s_diag_fired = 0;
            if (!s_diag_fired && sd->comp_ok && sc_i->cpu_ok) {
                STEREO_LOG("[PRESENT DIAG] Attempt (w=%u h=%u sc=%p) — GPU->CPU readback + layer pixel compare.",
                           sc_i->app_width, sc_i->app_height, (void*)sc_i);
                VkResult rd = alt_cpu_readback(sd, sc_i, queue, wcount, wsems,
                                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                if (rd == VK_SUCCESS) {
                    StereoDiagResult res = stereo_diagnose_layer_compare(sd, sc_i);
                    if (res == STEREO_DIAG_BLACK) {
                        /* Pure black frame → NOT a valid comparison.
                         * RE-ARM inner gate + outer gate → retry next present. */
                        STEREO_LOG("[PRESENT DIAG] BLACK frame detected — RE-ARMING both gates (retry on next frame with real scene).");
                        stereo_diagnose_layer_compare_set_done(0);
                        s_diag_fired = 0;
                    } else if (res == STEREO_DIAG_SKIPPED) {
                        if (sc_i->app_width >= 800 && sc_i->app_height >= 600) {
                            /* Size ok but compare refused for another reason —
                             * treat as consumed to avoid log spam. */
                            s_diag_fired = 1;
                        } else {
                            STEREO_LOG("[PRESENT DIAG] Frame too small — attempt not consumed.");
                        }
                    } else {
                        /* STEREO_DIAG_COMPLETED — real content, lock forever. */
                        s_diag_fired = 1;
                    }
                } else {
                    STEREO_LOG("[PRESENT DIAG] alt_cpu_readback FAILED res=%d — skipping this frame.",
                               (int)rd);
                }
            }
            /* Normal GPU blit compose for every frame.  (The diagnostic above
             * does not display anything — this function is always the real
             * screen update, so there is no GDI black-screen risk.) */
            pr = gpu_compose_present(sd, sc_i, queue, wcount, wsems);
            break;
        }
        default:
            pr = VK_SUCCESS;
            break;
        }
        if (pr != VK_SUCCESS) {
            const char *vk_err_name =
                (pr == VK_ERROR_OUT_OF_DATE_KHR) ? "OUT_OF_DATE_KHR (window rebuilt / resize — safe)" :
                (pr == VK_SUBOPTIMAL_KHR)        ? "SUBOPTIMAL_KHR (capability mismatch — safe)" :
                (pr == VK_ERROR_SURFACE_LOST_KHR)? "SURFACE_LOST_KHR (surface invalid)" :
                (pr == VK_ERROR_DEVICE_LOST)     ? "DEVICE_LOST (GPU hang / crash!)" :
                (pr == VK_ERROR_OUT_OF_HOST_MEMORY) ? "OUT_OF_HOST_MEMORY" :
                (pr == VK_ERROR_OUT_OF_DEVICE_MEMORY) ? "OUT_OF_DEVICE_MEMORY" : "UNKNOWN";
            STEREO_ERR("Present (mode=%d (%s)) failed: %d (%s). sc=%p w=%u h=%u hwnd=%p real_sc=%p",
                       (int)sc_i->present_mode,
                       (sc_i->present_mode == STEREO_PRESENT_SBS) ? "SBS" :
                       (sc_i->present_mode == STEREO_PRESENT_TAB) ? "TAB" :
                       (sc_i->present_mode == STEREO_PRESENT_INTERLACED) ? "INTERLACED" : "?",
                       pr, vk_err_name,
                       (void*)sc_i, sc_i->app_width, sc_i->app_height,
                       (void*)(uintptr_t)sc_i->hwnd,
                       (void*)(uintptr_t)sc_i->real_swapchain);
            result = pr;
        }
    }
    return result;
}

/* ── stereo_CreateImage / stereo_CreateImageView ────────────────────────── */

static bool is_depth_format(VkFormat fmt)
{
    switch (fmt) {
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:   case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default: return false;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateImage(VkDevice device, const VkImageCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
    STEREO_LOG_ONCE("FIRST_CALL stereo_CreateImage extent=%ux%u layers=%u fmt=%u",
        (unsigned)pCreateInfo->extent.width,
        (unsigned)pCreateInfo->extent.height,
        (unsigned)pCreateInfo->arrayLayers,
        (unsigned)pCreateInfo->format);
    STEREO_LOG_VERBOSE("CALLED stereo_CreateImage");
    static uint64_t image_create_seq = 0;
    uint64_t seq = ++image_create_seq;
    StereoDevice *sd = stereo_device_from_handle(device);
    STEREO_LOG_VERBOSE(
        "IMAGE device=%p sd=%p real=%p",
        (void*)device,
        (void*)sd,
        sd ? (void*)sd->real_device : NULL);
    STEREO_LOG_VERBOSE(
        "IMG_ENTER usage=0x%08X fmt=%u layers=%u samples=%u",
        pCreateInfo->usage,
        pCreateInfo->format,
        pCreateInfo->arrayLayers,
        pCreateInfo->samples);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    /* Upgrade images used as color/depth attachments (G-buffer and scene depth)
     * so multiview pipelines and framebuffers align.  This uses usage flags
     * rather than strictly matching the swapchain extent, preventing the
     * per-pass MV mismatch that caused overlapping geometry.
     */
    bool base = sd->stereo.enabled && sd->stereo.multiview
        && pCreateInfo
        && pCreateInfo->imageType   == VK_IMAGE_TYPE_2D
        && pCreateInfo->arrayLayers == 1;
    /* Depth/stencil attachments — upgraded for multiview depth per eye */
    bool intercept_depth = base
        && (pCreateInfo->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    /* Color attachments that are also sampled (render-to-texture G-buffers,
     * shadow-color, lighting output, post-fx targets).  mipLevels==1 and
     * extent > 1x1 to avoid upgrading LUTs or procedural textures.
     * Also intercept non-sampled color attachments (needed so every
     * framebuffer attachment has 2 layers for the multiview render pass). */
    bool intercept_color = base
        && pCreateInfo->mipLevels == 1
        && pCreateInfo->extent.width  > 1
        && pCreateInfo->extent.height > 1
        && (pCreateInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    STEREO_LOG_VERBOSE(
        "IMAGE_CREATE imageType=%d fmt=%d samples=%d usage=0x%x layers=%u extent=%ux%u flags=0x%x cube=%d array=%d upgrade=%d",
        pCreateInfo->imageType,
        pCreateInfo->format,
        pCreateInfo->samples,
        pCreateInfo->usage,
        pCreateInfo->arrayLayers,
        pCreateInfo->extent.width,
        pCreateInfo->extent.height,
        pCreateInfo->flags,
        !!(pCreateInfo->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT),
        pCreateInfo->arrayLayers > 1,
        intercept_depth || intercept_color);
    if (base &&
        (pCreateInfo->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
        !(pCreateInfo->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
    {
        STEREO_LOG_VERBOSE(
            "DEPTH_CREATE usage=0x%08X fmt=%u extent=%ux%u intercept_depth=%u intercept_color=%u",
            pCreateInfo->usage,
            pCreateInfo->format,
            pCreateInfo->extent.width,
            pCreateInfo->extent.height,
            intercept_depth,
            intercept_color);
    }
    if (!intercept_depth && !intercept_color)
    {
        STEREO_LOG_VERBOSE("CALL real CreateImage");
        VkResult r =
            sd->real.CreateImage(
            sd->real_device,
            pCreateInfo,
            pAllocator,
            pImage);
        STEREO_LOG_VERBOSE("RETURN real CreateImage result=%d", r);
        STEREO_LOG_VERBOSE(
            "IMG_EXIT passthrough result=%d image=%p",
            r,
            (r == VK_SUCCESS) ? (void *)(uintptr_t)*pImage : NULL);
        return r;
    }
    STEREO_LOG_VERBOSE(
        "IMAGE_UPGRADE usage=0x%08X fmt=%u extent=%ux%u depth=%u color=%u layers %u->2",
        pCreateInfo->usage,
        pCreateInfo->format,
        pCreateInfo->extent.width,
        pCreateInfo->extent.height,
        intercept_depth,
        intercept_color,
        pCreateInfo->arrayLayers);
    VkImageCreateInfo modified = *pCreateInfo;
    modified.arrayLayers = 2;
    STEREO_LOG_VERBOSE("CALL real CreateImage");
    VkResult res = sd->real.CreateImage(sd->real_device, &modified, pAllocator, pImage);
    STEREO_LOG_VERBOSE("RETURN real CreateImageView result=%d", res);
    STEREO_LOG(
        "[IMG UPGRADE] image=%p extent=%ux%u fmt=%u usage=0x%x layers=1->2 "
        "(depth=%d color=%d). This image is now tracked for double-layer blit interception.",
        (res == VK_SUCCESS) ? (void *)(uintptr_t)*pImage : NULL,
        pCreateInfo->extent.width,
        pCreateInfo->extent.height,
        pCreateInfo->format,
        pCreateInfo->usage,
        (int)intercept_depth,
        (int)intercept_color);
    if (res == VK_SUCCESS) {
        if (intercept_depth &&
            sd->intercepted_depth_count < MAX_DEPTH_IMAGES)
        {
            
            bool already_tracked = false;
            
            for (uint32_t i = 0;
                 i < sd->intercepted_depth_count;
                 i++)
            {
                if (sd->intercepted_depth[i] == *pImage)
                {
                    already_tracked = true;
            
            
                    break;
                }
            }
            
            if (!already_tracked)
            {
                STEREO_LOG_VERBOSE(
                    "DEPTH_INTERCEPT image=%p usage=0x%08X samples=%u fmt=%u",
                    (void *)(uintptr_t)*pImage,
                    pCreateInfo->usage,
                    pCreateInfo->samples,
                    pCreateInfo->format);
                if (sd->intercepted_depth_count < MAX_DEPTH_IMAGES)
                {
                    sd->intercepted_depth[
                        sd->intercepted_depth_count++] = *pImage;
                    STEREO_LOG_VERBOSE(
                        "DEPTH_TRACK count=%u image=%p",
                        sd->intercepted_depth_count,
                        (void *)(uintptr_t)*pImage);
                    STEREO_LOG_VERBOSE(
                        "COUNTS depth=%u color=%u upgradedImages=%u upgradedViews=%u",
                        sd->intercepted_depth_count,
                        sd->intercepted_color_count,
                        sd->upgraded_image_count,
                        sd->upgraded_view_count);
                }
            
            }
        }
        else if (intercept_depth)
        {
        }
        if (intercept_color &&
            sd->intercepted_color_count < MAX_COLOR_IMAGES)
        {
        bool already_tracked = false;
        for (uint32_t i = 0;
             i < sd->intercepted_color_count;
             i++)
        {
            if (sd->intercepted_color[i] == *pImage)
            {
                already_tracked = true;
                break;
            }
        }
        if (!already_tracked)
        {
            STEREO_LOG_VERBOSE(
                "COLOR_INTERCEPT image=%p extent=%ux%u mip=%u usage=0x%08X samples=%u fmt=%u",
                (void *)(uintptr_t)*pImage,
                pCreateInfo->extent.width,
                pCreateInfo->extent.height,
                pCreateInfo->mipLevels,
                pCreateInfo->usage,
                pCreateInfo->samples,
                pCreateInfo->format);
            if (sd->intercepted_color_count < MAX_COLOR_IMAGES)
            {
                sd->intercepted_color[
                    sd->intercepted_color_count++] = *pImage;
                STEREO_LOG_VERBOSE(
                    "COLOR_TRACK count=%u image=%p",
                    sd->intercepted_color_count,
                    (void *)(uintptr_t)*pImage);
                STEREO_LOG_VERBOSE(
                    "COUNTS depth=%u color=%u upgradedImages=%u upgradedViews=%u",
                    sd->intercepted_depth_count,
                    sd->intercepted_color_count,
                    sd->upgraded_image_count,
                    sd->upgraded_view_count);
            }
        }
        }
        else if (intercept_color)
        {
        }
    }
    return res;
}

VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateImageView(VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
    InterlockedIncrement((volatile long*)&g_stat_iv_total);
    STEREO_LOG_VERBOSE("CALLED stereo_CreateImageView");
    StereoDevice *sd = stereo_device_from_handle(device);
    STEREO_LOG_VERBOSE(
        "IV_ENTER image=%p viewType=%u layers=%u aspect=0x%X",
        (void *)(uintptr_t)pCreateInfo->image,
        pCreateInfo->viewType,
        pCreateInfo->subresourceRange.layerCount,
        pCreateInfo->subresourceRange.aspectMask);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    /*
     * Cube and cube-array images use array layers for faces.
     * They are not stereo render targets and must never be converted
     * into 2D array multiview views.
     */
    if (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE ||
        pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
    {
        STEREO_LOG_VERBOSE(
            "VIEW_SKIP cube-compatible viewType=%u layers=%u",
            pCreateInfo->viewType,
            pCreateInfo->subresourceRange.layerCount);
        return sd->real.CreateImageView(
            sd->real_device,
            pCreateInfo,
            pAllocator,
            pView);
    }
    if (!sd->stereo.multiview)
        return sd->real.CreateImageView(sd->real_device, pCreateInfo, pAllocator, pView);
    bool needs_upgrade = false;
    bool swapchain_match = false;
    bool depth_match = false;
    bool color_match = false;
    for (uint32_t si = 0; si < sd->swapchain_count; si++) {
        StereoSwapchain *scc = &sd->swapchains[si];
        if (!scc->stereo_active || !scc->stereo_images) continue;
        for (uint32_t ii = 0; ii < scc->image_count; ii++)
            if (scc->stereo_images[ii] == pCreateInfo->image)
            {
                needs_upgrade = true;
                swapchain_match = true;
            }
    }
    uint32_t depth_matches = 0;
    uint32_t color_matches = 0;
    for (uint32_t i = 0; i < sd->intercepted_depth_count; i++)
    {
        if (sd->intercepted_depth[i] == pCreateInfo->image)
        {
            depth_matches++;
            needs_upgrade = true;
            swapchain_match = true;
        }
    }
    for (uint32_t i = 0; i < sd->intercepted_color_count; i++)
    {
        if (sd->intercepted_color[i] == pCreateInfo->image)
        {
            color_matches++;
            needs_upgrade = true;
            swapchain_match = true;
        }
    }
    for (uint32_t i = 0; i < sd->upgraded_image_count; i++)
    {
        if (sd->upgraded_images[i] == pCreateInfo->image)
        {
            color_matches++;
            needs_upgrade = true;
            swapchain_match = true;
        }
    }
    if (!needs_upgrade &&
        (pCreateInfo->subresourceRange.aspectMask &
         VK_IMAGE_ASPECT_DEPTH_BIT))
    {
    }
    if (!needs_upgrade)
       {
        static volatile long s_passthrough_tick = 0;
        long tick = InterlockedIncrement(&s_passthrough_tick);
        if (tick % 500 == 1) {
            STEREO_LOG(
                "IV_SAMPLE[%ld] PASSTHROUGH image=%p fmt=%u vt=%u layers=%u aspect=0x%X "
                "total_so_far=%u",
                tick,
                (void*)(uintptr_t)pCreateInfo->image,
                (unsigned)pCreateInfo->format,
                (unsigned)pCreateInfo->viewType,
                (unsigned)pCreateInfo->subresourceRange.layerCount,
                (unsigned)pCreateInfo->subresourceRange.aspectMask,
                (unsigned)g_stat_iv_total);
        }
        STEREO_LOG_VERBOSE(
            "VIEW_PASSTHROUGH image=%p fmt=%u aspect=0x%X viewType=%u layers=%u depthTracked=%u colorTracked=%u usage_unknown=1",
            (void*)(uintptr_t)pCreateInfo->image,
            pCreateInfo->format,
            pCreateInfo->subresourceRange.aspectMask,
            pCreateInfo->viewType,
            pCreateInfo->subresourceRange.layerCount,
            sd->intercepted_depth_count,
            sd->intercepted_color_count);
        STEREO_LOG_VERBOSE("CALL real CreateImageView");
        VkResult r =
            sd->real.CreateImageView(
            sd->real_device,
            pCreateInfo,
            pAllocator,
            pView);
        STEREO_LOG_VERBOSE("RETURN real CreateImageView result=%d", r);
        STEREO_LOG_VERBOSE(
            "IV_EXIT passthrough result=%d view=%p",
            r,
            (r == VK_SUCCESS) ? (void *)(uintptr_t)*pView : NULL);
        return r;
       }
    STEREO_LOG_VERBOSE(
        "UPGRADE_REASON image=%p swapchain=%u depth=%u color=%u",
        (void *)(uintptr_t)pCreateInfo->image,
        swapchain_match,
        depth_match,
        color_match);
    /* Multiview requires framebuffer attachments to be 2D_ARRAY with
     * layerCount >= popcount(viewMask). Upgrade 2D→2D_ARRAY, layers 1→2. */
    VkImageViewCreateInfo upgraded = *pCreateInfo;
    if (upgraded.viewType == VK_IMAGE_VIEW_TYPE_2D)
        upgraded.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    if (upgraded.subresourceRange.layerCount < 2)
        upgraded.subresourceRange.layerCount = 2;
    STEREO_LOG_VERBOSE(
        "VIEW_UPGRADE image=%p fmt=%u oldType=%u newType=%u oldLayers=%u newLayers=%u",
        (void*)(uintptr_t)pCreateInfo->image,
        pCreateInfo->format,
        pCreateInfo->viewType,
        upgraded.viewType,
        pCreateInfo->subresourceRange.layerCount,
        upgraded.subresourceRange.layerCount);
    VkResult _r = sd->real.CreateImageView(
        sd->real_device, &upgraded, pAllocator, pView);
    if (_r == VK_SUCCESS) {
        static volatile long s_upgrade_tick = 0;
        long tick = InterlockedIncrement(&s_upgrade_tick);
        InterlockedIncrement((volatile long*)&g_stat_iv_upgraded);
        if (tick % 250 == 1) {
            STEREO_LOG(
                "IV_SAMPLE[%ld] UPGRADED view=%p image=%p vt=%u->%u layers=%u->%u "
                "total_so_far=%u upgraded=%u",
                tick,
                (void*)(uintptr_t)*pView,
                (void*)(uintptr_t)pCreateInfo->image,
                (unsigned)pCreateInfo->viewType,
                (unsigned)upgraded.viewType,
                (unsigned)pCreateInfo->subresourceRange.layerCount,
                (unsigned)upgraded.subresourceRange.layerCount,
                (unsigned)g_stat_iv_total,
                (unsigned)g_stat_iv_upgraded);
        }
    }
    if (_r == VK_SUCCESS &&
        sd->upgraded_view_count < MAX_UPGRADED_VIEWS)
    {
        CHECK_ARRAY_COUNT(sd->upgraded_view_count, MAX_UPGRADED_VIEWS, "upgraded_view_count");
        sd->upgraded_views[sd->upgraded_view_count++] = *pView;
        if (sd->upgraded_image_count < MAX_UPGRADED_VIEWS)
        {
            CHECK_ARRAY_COUNT(sd->upgraded_image_count, MAX_UPGRADED_VIEWS, "upgraded_image_count");
            sd->upgraded_images[sd->upgraded_image_count++] = pCreateInfo->image;
        }
    }
    return _r;
}

VKAPI_ATTR void VKAPI_CALL
stereo_DestroyImageView(
    VkDevice device,
    VkImageView imageView,
    const VkAllocationCallbacks *pAllocator)
{
    STEREO_LOG_VERBOSE("CALLED stereo_DestroyImageView");
    STEREO_LOG_VERBOSE(
        "DestroyImageView %p",
        (void *)(uintptr_t)imageView);
    //STEREO_LOG_VERBOSE(
    //    "[DESTROY IMAGEVIEW ENTRY] view=%p",
    //    imageView);
    //STEREO_LOG_VERBOSE(
    //    "[VIEW DESTROY ENTRY] view=%p",
    //    imageView);
    StereoDevice *sd = stereo_device_from_handle(device);
    if (!sd)
        return;
    //STEREO_LOG_VERBOSE(
    //    "[VIEW DESTROY LOOKUP] count=%u",
    //    sd->upgraded_view_count);

    for (uint32_t i = 0;
         i < sd->upgraded_view_count;
         i++)
    {
        if (sd->upgraded_views[i] == imageView)
        {
            //STEREO_LOG_VERBOSE(
            //    "[VIEW TRACK REMOVE] view=%p slot=%u",
            //    imageView,
            //    i);
            uint32_t last = --sd->upgraded_view_count;
            if (i != last)
                sd->upgraded_views[i] = sd->upgraded_views[last];
            sd->upgraded_views[last] = VK_NULL_HANDLE;
            //STEREO_LOG_VERBOSE(
            //    "[VIEW TRACK COUNT] count=%u",
            //    sd->upgraded_view_count);
            break;
        }
    }
    //STEREO_LOG_VERBOSE(
    //    "[VIEW DESTROY MISS] view=%p count=%u",
    //    imageView,
    //    sd->upgraded_view_count);
    sd->real.DestroyImageView(
        sd->real_device,
        imageView,
        pAllocator);
}