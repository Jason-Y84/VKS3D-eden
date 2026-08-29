/*
 * image_blit.c — Image transfer interception for Yuzu (Switch emulator)
 *
 * Root cause of the "right eye black screen" on Yuzu:
 *   Yuzu does not render directly into the swapchain image.  Its
 *   presentation pipeline is:
 *
 *     1. Create swapchain image → VKS3D intercepts and returns
 *        stereo_images[0] with arrayLayers=2.
 *     2. Allocate independent intermediate color attachments via
 *        vkCreateImage → VKS3D upgrades these to arrayLayers=2 as
 *        well (stereo_CreateImage in swapchain.c).
 *     3. Render the frame into the intermediate RT using dynamic
 *        rendering with multiview viewMask=0x3 → VS patch writes
 *        layer 0 (left eye) and layer 1 (right eye).
 *     4. vkCmdBlitImage or vkCmdCopyImage from the intermediate RT
 *        to the swapchain image.  ← THIS WAS NOT INTERCEPTED.
 *
 *   Without wrappers here, the Blit/Copy issued by Yuzu only copies
 *   layer 0 of the source to layer 0 of the destination — layer 1
 *   of stereo_images[0] is never written and remains in
 *   VK_IMAGE_LAYOUT_UNDEFINED.  gpu_compose_present then tries to
 *   blit layer 1 of stereo_images[0] into the right half of the SBS
 *   canvas, producing a completely black right eye.
 *
 * The fix is simple: whenever the source OR the destination image
 * is tracked as a 2-layer upgraded image, issue TWO real transfer
 * commands.  The first one replicates the original region (keeping
 * baseArrayLayer = 0 on both sides).  The second one adjusts the
 * regions to target baseArrayLayer = 1 on both sides.
 *
 * If both source and destination are plain (not upgraded) images,
 * we pass through the call unchanged — no extra cost for regular
 * PC games that render directly to the swapchain image.
 */

#include <stdio.h>
#include <string.h>
#include "stereo_icd.h"

/* -- Hardened build diagnostics --------------------------------------------- *
 * The 15 public stereo_Cmd*Image*() wrappers defined in this TU are the ONLY
 * definitions of their symbols in the entire DLL.  Any condition that
 * prevents this translation unit from reaching its 15 definitions will
 * surface as 15 LNK2019 in the final link step — exactly the failure we
 * saw once this file was pushed to GitHub CI.
 *
 * We therefore pin down every prerequisite here, IN THIS FILE, so a
 * broken CI pipeline fails EARLY with an actionable error instead of
 * silently dropping the TU.
 *
 * Checklist of things that can kill this TU on an arbitrary machine:
 *   1. <vulkan/vulkan.h> was not found at all
 *   2. Vulkan SDK is pre-1.3 and has no VK_KHR_synchronization2 types
 *      → stereo_icd.h now ships its own fallback shims (gated by
 *         !defined(VK_KHR_synchronization2)), so this is covered.
 *   3. sd->real dispatch table layout drifts relative to the member
 *      names we write to here (CmdBlitImage / CmdBlitImage2 / ...).
 *   4. Helper helper helpers helper helpers helper helper helpers.
 * ------------------------------------------------------------------------ */
#if !defined(__has_include)
  /* Legacy compilers: just keep going, the #include above already
   * either succeeded or died with "cannot open include file". */
#elif !__has_include(<vulkan/vulkan.h>)
#  error "image_blit.c: <vulkan/vulkan.h> not found. Install Vulkan SDK 1.3+ or set VULKAN_SDK env var."
#endif

#if !defined(VK_VERSION_1_0)
#  error "image_blit.c: Vulkan core types not visible — check include order and SDK installation."
#endif

/* Confirm stereo_icd.h successfully provided every *2 sync2 type the
 * functions below rely on.  If any of these fire, the SDK shim block
 * in stereo_icd.h needs an extra entry (or the SDK is really broken). */
#if !defined(VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2)
#  error "image_blit.c: VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2 missing — please update stereo_icd.h Sync2 shims or use Vulkan SDK >= 1.3.202"
#endif
#if !defined(VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2)
#  error "image_blit.c: VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2 missing — please update stereo_icd.h Sync2 shims or use Vulkan SDK >= 1.3.202"
#endif
#if !defined(VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2)
#  error "image_blit.c: VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2 missing — please update stereo_icd.h Sync2 shims or use Vulkan SDK >= 1.3.202"
#endif
#if !defined(VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2)
#  error "image_blit.c: VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2 missing — please update stereo_icd.h Sync2 shims or use Vulkan SDK >= 1.3.202"
#endif
#if !defined(VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2)
#  error "image_blit.c: VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2 missing — please update stereo_icd.h Sync2 shims or use Vulkan SDK >= 1.3.202"
#endif

/* Confirm every required field of RealDeviceDispatch is present at
 * compile time.  sizeof(sd->real.FIELD) on a pointer-to-member yields
 * the size of the PFN typedef — non-zero if the member exists, hard
 * compile error if it was renamed or removed.  Uses the standard C
 * `offsetof` / explicit type-size verification via _Static_assert so
 * this works as plain C11 (no C++ required). */
#define IMAGE_BLIT_ASSERT_MEMBER(Struct, Field) \
    _Static_assert(sizeof(((Struct *)0)->Field) > 0, \
        "RealDeviceDispatch missing required field ` " #Field " `")
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdBlitImage);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdCopyImage);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdResolveImage);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdCopyBufferToImage);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdCopyImageToBuffer);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdBlitImage2);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdCopyImage2);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdResolveImage2);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdCopyBufferToImage2);
IMAGE_BLIT_ASSERT_MEMBER(RealDeviceDispatch, CmdCopyImageToBuffer2);
#undef IMAGE_BLIT_ASSERT_MEMBER


/* Stack allocation cap for transfer regions.  Real Vulkan apps almost
 * never pass more than ~8 regions per call; 128 is a conservative cap
 * that lets us avoid the malloc/free overhead entirely in 99 % of
 * frames.  Anything larger falls back to heap allocation. */
#define BLIT_STACK_REGIONS  (128U)

/* ────────────────────────────────────────────────────────────────────── *
 *  Device lookup (mirrors find_any_device in framebuffer.c)
 * ────────────────────────────────────────────────────────────────────── */
static StereoDevice *
find_any_device_blit(void)
{
    extern StereoDevice g_devices[];
    extern uint32_t g_device_count;

    for (uint32_t i = 0; i < g_device_count; i++)
    {
        if (g_devices[i].real_device)
            return &g_devices[i];
    }
    return NULL;
}

/* ────────────────────────────────────────────────────────────────────── *
 *  One-shot per-(src,dst) log deduplication
 *
 *  Each unique (src_image, dst_image) pair is logged exactly once at
 *  STEREO_LOG level (visible even with VERBOSE=0).  Subsequent calls
 *  with the same pair fall through to STEREO_LOG_VERBOSE so we don't
 *  spam the log every frame but still keep a trace for full debug.
 * ────────────────────────────────────────────────────────────────────── */
#define BLIT_DEDUP_CAP    64

typedef struct {
    uint64_t src;
    uint64_t dst;
} BlitDedupKey;

static BlitDedupKey s_blit_seen[BLIT_DEDUP_CAP];
static uint32_t     s_blit_seen_count = 0;

/* Returns true if this is the first time we see this (src,dst) pair,
 * and records it so subsequent calls return false.  Thread-unsafe but
 * blit commands are always recorded within a single command buffer
 * (single-threaded recording context per Vulkan spec). */
static bool blit_log_first_time(VkImage src, VkImage dst)
{
    uint64_t s = (uint64_t)(uintptr_t)src;
    uint64_t d = (uint64_t)(uintptr_t)dst;
    for (uint32_t i = 0; i < s_blit_seen_count; i++)
    {
        if (s_blit_seen[i].src == s && s_blit_seen[i].dst == d)
            return false;
    }
    if (s_blit_seen_count < BLIT_DEDUP_CAP)
    {
        s_blit_seen[s_blit_seen_count].src = s;
        s_blit_seen[s_blit_seen_count].dst = d;
        s_blit_seen_count++;
    }
    return true;
}

/* Pretty-print which tracked list the image belongs to.  Useful for
 * reading the diagnostic log without cross-referencing pointers. */
static const char *
image_layered_origin(StereoDevice *sd, VkImage image, bool *is_intercepted_color,
                     bool *is_intercepted_depth, bool *is_upgraded_image)
{
    *is_intercepted_color = false;
    *is_intercepted_depth = false;
    *is_upgraded_image    = false;
    if (!sd || image == VK_NULL_HANDLE)
        return "null";

    for (uint32_t i = 0; i < sd->intercepted_color_count; i++)
    {
        if (sd->intercepted_color[i] == image)
        {
            *is_intercepted_color = true;
            return "intercepted_color (stereo swapchain image)";
        }
    }
    for (uint32_t i = 0; i < sd->intercepted_depth_count; i++)
    {
        if (sd->intercepted_depth[i] == image)
        {
            *is_intercepted_depth = true;
            return "intercepted_depth";
        }
    }
    for (uint32_t i = 0; i < sd->upgraded_image_count; i++)
    {
        if (sd->upgraded_images[i] == image)
        {
            *is_upgraded_image = true;
            return "upgraded_image (CreateImage 2-layer upgrade)";
        }
    }
    return "non-tracked (single layer)";
}

/* Helper: log first-time blit/copy call at STEREO_LOG level.  Don't try
 * to parse region contents — struct layouts differ across Blit/Copy/Resolve
 * and their Vk*2 variants, so wrong offsets would yield misleading logs.
 * The (src,dst) deduplication prevents per-frame spam. */
static void blit_log_layers(const char *fn, VkImage src, VkImage dst,
                            bool src_upg, bool dst_upg,
                            uint32_t regionCount, const void *pRegions,
                            bool is_src_layered2, bool is_dst_layered2)
{
    (void)pRegions;
    (void)is_src_layered2;
    (void)is_dst_layered2;

    bool first = blit_log_first_time(src, dst);
    if (!first)
    {
        STEREO_LOG_VERBOSE(
            "%s src=%p dst=%p regions=%u src_upg=%u dst_upg=%u (already logged)",
            fn, (void*)(uintptr_t)src, (void*)(uintptr_t)dst,
            regionCount, src_upg, dst_upg);
        return;
    }

    StereoDevice *sd = find_any_device_blit();
    bool src_is_color = false, src_is_depth = false, src_is_upg = false;
    bool dst_is_color = false, dst_is_depth = false, dst_is_upg = false;
    const char *src_origin = image_layered_origin(sd, src,
                                                   &src_is_color, &src_is_depth, &src_is_upg);
    const char *dst_origin = image_layered_origin(sd, dst,
                                                   &dst_is_color, &dst_is_depth, &dst_is_upg);

    STEREO_LOG(
        "[BLIT TRACK] %s FIRST-TIME src=%p(%s, upg=%d) dst=%p(%s, upg=%d) "
        "regions=%u  => %s",
        fn,
        (void*)(uintptr_t)src, src_origin, (int)src_upg,
        (void*)(uintptr_t)dst, dst_origin, (int)dst_upg,
        regionCount,
        (src_upg || dst_upg) ? "DOUBLE-LAYER COPY (layer 0 + layer 1)"
                              : "passthrough single-layer");
}

/* Same as blit_log_layers but for buffer<->image commands (no dst/src image). */
static void blit_log_buffer_image(const char *fn, VkImage img, bool img_upg,
                                  uint32_t regionCount, const void *pRegions)
{
    (void)pRegions;

    bool first = blit_log_first_time(img, (VkImage)0);
    if (!first)
    {
        STEREO_LOG_VERBOSE(
            "%s img=%p upg=%u regions=%u (already logged)",
            fn, (void*)(uintptr_t)img, img_upg, regionCount);
        return;
    }

    StereoDevice *sd = find_any_device_blit();
    bool is_color = false, is_depth = false, is_upg = false;
    const char *origin = image_layered_origin(sd, img, &is_color, &is_depth, &is_upg);

    STEREO_LOG(
        "[BLIT TRACK] %s FIRST-TIME img=%p(%s, upg=%d) regions=%u  => %s",
        fn,
        (void*)(uintptr_t)img, origin, (int)img_upg,
        regionCount,
        img_upg ? "DOUBLE-LAYER COPY (layer 0 + layer 1)"
                : "passthrough single-layer");
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_image_is_layered2
 *
 *  Returns true if `image` is tracked in the StereoDevice as one of the
 *  2-layer upgraded images: a swapchain stereo image (intercepted_color),
 *  an intercepted depth attachment, or an arbitrary image whose
 *  arrayLayers got upgraded in stereo_CreateImage.
 * ────────────────────────────────────────────────────────────────────── */
bool
stereo_image_is_layered2(StereoDevice *sd, VkImage image)
{
    if (!sd || image == VK_NULL_HANDLE)
        return false;

    for (uint32_t i = 0; i < sd->intercepted_color_count; i++)
    {
        if (sd->intercepted_color[i] == image)
            return true;
    }
    for (uint32_t i = 0; i < sd->intercepted_depth_count; i++)
    {
        if (sd->intercepted_depth[i] == image)
            return true;
    }
    for (uint32_t i = 0; i < sd->upgraded_image_count; i++)
    {
        if (sd->upgraded_images[i] == image)
            return true;
    }
    return false;
}

/* Check if an image is specifically a stereo swapchain color image
 * (i.e. stereo_images[0] or equivalent).  Used to decide whether to
 * force a layout fix-back-to-COLOR_ATTACHMENT_OPTIMAL after Eden's
 * blit completes.  This excludes depth attachments and upgraded RT
 * images — only the final stereo output image qualifies. */
bool
stereo_image_is_intercepted_color(StereoDevice *sd, VkImage image)
{
    if (!sd || image == VK_NULL_HANDLE)
        return false;
    for (uint32_t i = 0; i < sd->intercepted_color_count; i++)
    {
        if (sd->intercepted_color[i] == image)
            return true;
    }
    return false;
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdBlitImage
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdBlitImage(
    VkCommandBuffer      commandBuffer,
    VkImage              srcImage,
    VkImageLayout        srcImageLayout,
    VkImage              dstImage,
    VkImageLayout        dstImageLayout,
    uint32_t             regionCount,
    const VkImageBlit   *pRegions,
    VkFilter             filter)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdBlitImage)
        return;
    if (regionCount == 0 || pRegions == NULL)
    {
        sd->real.CmdBlitImage(commandBuffer, srcImage, srcImageLayout,
                              dstImage, dstImageLayout,
                              0, NULL, filter);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, srcImage);
    const bool dst_upg = stereo_image_is_layered2(sd, dstImage);

    blit_log_layers("CmdBlitImage", srcImage, dstImage,
                    src_upg, dst_upg, regionCount, pRegions,
                    src_upg, dst_upg);

    if (!src_upg && !dst_upg)
    {
        sd->real.CmdBlitImage(commandBuffer, srcImage, srcImageLayout,
                              dstImage, dstImageLayout,
                              regionCount, pRegions, filter);
        return;
    }

    /* Layer 0 blit: keep the original regions as-is. */
    sd->real.CmdBlitImage(commandBuffer, srcImage, srcImageLayout,
                          dstImage, dstImageLayout,
                          regionCount, pRegions, filter);

    VkImageBlit  stack[BLIT_STACK_REGIONS];
    VkImageBlit *tmp = (regionCount <= BLIT_STACK_REGIONS) ? stack :
                       (VkImageBlit *)malloc(sizeof(VkImageBlit) * regionCount);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdBlitImage: OOM allocating %u regions", regionCount);
        return;
    }
    memcpy(tmp, pRegions, sizeof(VkImageBlit) * regionCount);

    for (uint32_t r = 0; r < regionCount; r++)
    {
        if (src_upg)
            tmp[r].srcSubresource.baseArrayLayer = 1;
        if (dst_upg)
            tmp[r].dstSubresource.baseArrayLayer = 1;
    }

    sd->real.CmdBlitImage(commandBuffer, srcImage, srcImageLayout,
                          dstImage, dstImageLayout,
                          regionCount, tmp, filter);
    if (tmp != stack)
        free(tmp);

    /* After Eden blits into stereo_images[0] (our SBS swapchain image),
     * the image's actual layout is dstImageLayout (typically
     * TRANSFER_DST_OPTIMAL or GENERAL), NOT COLOR_ATTACHMENT_OPTIMAL.
     * But alt_cpu_readback and gpu_compose_present both assume
     * oldLayout=COLOR_ATTACHMENT_OPTIMAL when transitioning the image
     * for their own readback/blit.  Mismatched oldLayout causes NVIDIA
     * driver to discard image content (DIAG readback shows all-black
     * even though the image actually has rendered content).
     *
     * Fix: insert a layout transition barrier to force stereo_images[0]
     * back to COLOR_ATTACHMENT_OPTIMAL right after Eden's blit completes.
     * This makes the downstream oldLayout=COLOR_ATTACHMENT_OPTIMAL in
     * present_alt.c match the actual image layout. */
    if (dst_upg && stereo_image_is_intercepted_color(sd, dstImage)) {
        VkImageMemoryBarrier fix = {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout           = dstImageLayout,
            .newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = dstImage,
            .subresourceRange    = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2
            },
        };
        sd->real.CmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 1, &fix);
        static bool s_layout_fix_logged = false;
        if (!s_layout_fix_logged) {
            s_layout_fix_logged = true;
            STEREO_LOG(
                "[BLIT LAYOUT FIX] dst=%p dstLayout=%u -> COLOR_ATTACHMENT_OPTIMAL "
                "(ensures downstream readback/gpu_compose oldLayout matches)",
                (void*)(uintptr_t)dstImage, (unsigned)dstImageLayout);
        }
    }
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdCopyImage
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyImage(
    VkCommandBuffer      commandBuffer,
    VkImage              srcImage,
    VkImageLayout        srcImageLayout,
    VkImage              dstImage,
    VkImageLayout        dstImageLayout,
    uint32_t             regionCount,
    const VkImageCopy   *pRegions)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdCopyImage)
        return;
    if (regionCount == 0 || pRegions == NULL)
    {
        sd->real.CmdCopyImage(commandBuffer, srcImage, srcImageLayout,
                              dstImage, dstImageLayout, 0, NULL);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, srcImage);
    const bool dst_upg = stereo_image_is_layered2(sd, dstImage);

    blit_log_layers("CmdCopyImage", srcImage, dstImage,
                    src_upg, dst_upg, regionCount, pRegions,
                    src_upg, dst_upg);

    if (!src_upg && !dst_upg)
    {
        sd->real.CmdCopyImage(commandBuffer, srcImage, srcImageLayout,
                              dstImage, dstImageLayout,
                              regionCount, pRegions);
        return;
    }

    sd->real.CmdCopyImage(commandBuffer, srcImage, srcImageLayout,
                          dstImage, dstImageLayout,
                          regionCount, pRegions);

    VkImageCopy  stack[BLIT_STACK_REGIONS];
    VkImageCopy *tmp = (regionCount <= BLIT_STACK_REGIONS) ? stack :
                       (VkImageCopy *)malloc(sizeof(VkImageCopy) * regionCount);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdCopyImage: OOM allocating %u regions", regionCount);
        return;
    }
    memcpy(tmp, pRegions, sizeof(VkImageCopy) * regionCount);

    for (uint32_t r = 0; r < regionCount; r++)
    {
        if (src_upg)
            tmp[r].srcSubresource.baseArrayLayer = 1;
        if (dst_upg)
            tmp[r].dstSubresource.baseArrayLayer = 1;
    }

    sd->real.CmdCopyImage(commandBuffer, srcImage, srcImageLayout,
                          dstImage, dstImageLayout,
                          regionCount, tmp);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdResolveImage
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdResolveImage(
    VkCommandBuffer       commandBuffer,
    VkImage               srcImage,
    VkImageLayout         srcImageLayout,
    VkImage               dstImage,
    VkImageLayout         dstImageLayout,
    uint32_t              regionCount,
    const VkImageResolve *pRegions)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdResolveImage)
        return;
    if (regionCount == 0 || pRegions == NULL)
    {
        sd->real.CmdResolveImage(commandBuffer, srcImage, srcImageLayout,
                                 dstImage, dstImageLayout, 0, NULL);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, srcImage);
    const bool dst_upg = stereo_image_is_layered2(sd, dstImage);

    blit_log_layers("CmdResolveImage", srcImage, dstImage,
                    src_upg, dst_upg, regionCount, pRegions,
                    src_upg, dst_upg);

    if (!src_upg && !dst_upg)
    {
        sd->real.CmdResolveImage(commandBuffer, srcImage, srcImageLayout,
                                 dstImage, dstImageLayout,
                                 regionCount, pRegions);
        return;
    }

    sd->real.CmdResolveImage(commandBuffer, srcImage, srcImageLayout,
                             dstImage, dstImageLayout,
                             regionCount, pRegions);

    VkImageResolve  stack[BLIT_STACK_REGIONS];
    VkImageResolve *tmp = (regionCount <= BLIT_STACK_REGIONS) ? stack :
                          (VkImageResolve *)malloc(sizeof(VkImageResolve) * regionCount);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdResolveImage: OOM allocating %u regions", regionCount);
        return;
    }
    memcpy(tmp, pRegions, sizeof(VkImageResolve) * regionCount);

    for (uint32_t r = 0; r < regionCount; r++)
    {
        if (src_upg)
            tmp[r].srcSubresource.baseArrayLayer = 1;
        if (dst_upg)
            tmp[r].dstSubresource.baseArrayLayer = 1;
    }

    sd->real.CmdResolveImage(commandBuffer, srcImage, srcImageLayout,
                             dstImage, dstImageLayout,
                             regionCount, tmp);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdCopyBufferToImage
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyBufferToImage(
    VkCommandBuffer         commandBuffer,
    VkBuffer                srcBuffer,
    VkImage                 dstImage,
    VkImageLayout           dstImageLayout,
    uint32_t                regionCount,
    const VkBufferImageCopy *pRegions)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdCopyBufferToImage)
        return;
    if (regionCount == 0 || pRegions == NULL)
    {
        sd->real.CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage,
                                      dstImageLayout, 0, NULL);
        return;
    }

    const bool dst_upg = stereo_image_is_layered2(sd, dstImage);

    blit_log_buffer_image("CmdCopyBufferToImage", dstImage, dst_upg,
                          regionCount, pRegions);

    if (!dst_upg)
    {
        sd->real.CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage,
                                      dstImageLayout,
                                      regionCount, pRegions);
        return;
    }

    sd->real.CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage,
                                  dstImageLayout,
                                  regionCount, pRegions);

    VkBufferImageCopy  stack[BLIT_STACK_REGIONS];
    VkBufferImageCopy *tmp = (regionCount <= BLIT_STACK_REGIONS) ? stack :
                            (VkBufferImageCopy *)malloc(sizeof(VkBufferImageCopy) * regionCount);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdCopyBufferToImage: OOM allocating %u regions", regionCount);
        return;
    }
    memcpy(tmp, pRegions, sizeof(VkBufferImageCopy) * regionCount);

    for (uint32_t r = 0; r < regionCount; r++)
    {
        tmp[r].imageSubresource.baseArrayLayer = 1;
    }

    sd->real.CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage,
                                  dstImageLayout,
                                  regionCount, tmp);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdCopyImageToBuffer (legacy)
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyImageToBuffer(
    VkCommandBuffer         commandBuffer,
    VkImage                 srcImage,
    VkImageLayout           srcImageLayout,
    VkBuffer                dstBuffer,
    uint32_t                regionCount,
    const VkBufferImageCopy *pRegions)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdCopyImageToBuffer)
        return;
    if (regionCount == 0 || pRegions == NULL)
    {
        sd->real.CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout,
                                      dstBuffer, 0, NULL);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, srcImage);

    blit_log_buffer_image("CmdCopyImageToBuffer", srcImage, src_upg,
                          regionCount, pRegions);

    if (!src_upg)
    {
        sd->real.CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout,
                                      dstBuffer,
                                      regionCount, pRegions);
        return;
    }

    sd->real.CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout,
                                  dstBuffer,
                                  regionCount, pRegions);

    VkBufferImageCopy  stack[BLIT_STACK_REGIONS];
    VkBufferImageCopy *tmp = (regionCount <= BLIT_STACK_REGIONS) ? stack :
                            (VkBufferImageCopy *)malloc(sizeof(VkBufferImageCopy) * regionCount);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdCopyImageToBuffer: OOM allocating %u regions", regionCount);
        return;
    }
    memcpy(tmp, pRegions, sizeof(VkBufferImageCopy) * regionCount);

    for (uint32_t r = 0; r < regionCount; r++)
    {
        tmp[r].imageSubresource.baseArrayLayer = 1;
    }

    sd->real.CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout,
                                  dstBuffer,
                                  regionCount, tmp);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdBlitImage2  (Sync2 / VK 1.3 core)
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdBlitImage2(
    VkCommandBuffer         commandBuffer,
    const VkBlitImageInfo2 *pInfo)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdBlitImage2)
        return;
    if (pInfo == NULL)
        return;
    if (pInfo->regionCount == 0 || pInfo->pRegions == NULL)
    {
        sd->real.CmdBlitImage2(commandBuffer, pInfo);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, pInfo->srcImage);
    const bool dst_upg = stereo_image_is_layered2(sd, pInfo->dstImage);

    blit_log_layers("CmdBlitImage2", pInfo->srcImage, pInfo->dstImage,
                    src_upg, dst_upg, pInfo->regionCount, pInfo->pRegions,
                    src_upg, dst_upg);

    if (!src_upg && !dst_upg)
    {
        sd->real.CmdBlitImage2(commandBuffer, pInfo);
        return;
    }

    sd->real.CmdBlitImage2(commandBuffer, pInfo);

    const uint32_t  n = pInfo->regionCount;
    VkImageBlit2    stack[BLIT_STACK_REGIONS];
    VkImageBlit2   *tmp = (n <= BLIT_STACK_REGIONS) ? stack :
                          (VkImageBlit2 *)malloc(sizeof(VkImageBlit2) * n);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdBlitImage2: OOM allocating %u regions", n);
        return;
    }
    memcpy(tmp, pInfo->pRegions, sizeof(VkImageBlit2) * n);

    for (uint32_t r = 0; r < n; r++)
    {
        if (src_upg)
            tmp[r].srcSubresource.baseArrayLayer = 1;
        if (dst_upg)
            tmp[r].dstSubresource.baseArrayLayer = 1;
    }

    VkBlitImageInfo2 info2 = *pInfo;
    info2.pRegions = tmp;
    sd->real.CmdBlitImage2(commandBuffer, &info2);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdCopyImage2  (Sync2 / VK 1.3 core)
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyImage2(
    VkCommandBuffer         commandBuffer,
    const VkCopyImageInfo2 *pInfo)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdCopyImage2)
        return;
    if (pInfo == NULL)
        return;
    if (pInfo->regionCount == 0 || pInfo->pRegions == NULL)
    {
        sd->real.CmdCopyImage2(commandBuffer, pInfo);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, pInfo->srcImage);
    const bool dst_upg = stereo_image_is_layered2(sd, pInfo->dstImage);

    blit_log_layers("CmdCopyImage2", pInfo->srcImage, pInfo->dstImage,
                    src_upg, dst_upg, pInfo->regionCount, pInfo->pRegions,
                    src_upg, dst_upg);

    if (!src_upg && !dst_upg)
    {
        sd->real.CmdCopyImage2(commandBuffer, pInfo);
        return;
    }

    sd->real.CmdCopyImage2(commandBuffer, pInfo);

    const uint32_t  n = pInfo->regionCount;
    VkImageCopy2    stack[BLIT_STACK_REGIONS];
    VkImageCopy2   *tmp = (n <= BLIT_STACK_REGIONS) ? stack :
                          (VkImageCopy2 *)malloc(sizeof(VkImageCopy2) * n);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdCopyImage2: OOM allocating %u regions", n);
        return;
    }
    memcpy(tmp, pInfo->pRegions, sizeof(VkImageCopy2) * n);

    for (uint32_t r = 0; r < n; r++)
    {
        if (src_upg)
            tmp[r].srcSubresource.baseArrayLayer = 1;
        if (dst_upg)
            tmp[r].dstSubresource.baseArrayLayer = 1;
    }

    VkCopyImageInfo2 info2 = *pInfo;
    info2.pRegions = tmp;
    sd->real.CmdCopyImage2(commandBuffer, &info2);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdResolveImage2  (Sync2 / VK 1.3 core)
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdResolveImage2(
    VkCommandBuffer            commandBuffer,
    const VkResolveImageInfo2 *pInfo)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdResolveImage2)
        return;
    if (pInfo == NULL)
        return;
    if (pInfo->regionCount == 0 || pInfo->pRegions == NULL)
    {
        sd->real.CmdResolveImage2(commandBuffer, pInfo);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, pInfo->srcImage);
    const bool dst_upg = stereo_image_is_layered2(sd, pInfo->dstImage);

    blit_log_layers("CmdResolveImage2", pInfo->srcImage, pInfo->dstImage,
                    src_upg, dst_upg, pInfo->regionCount, pInfo->pRegions,
                    src_upg, dst_upg);

    if (!src_upg && !dst_upg)
    {
        sd->real.CmdResolveImage2(commandBuffer, pInfo);
        return;
    }

    sd->real.CmdResolveImage2(commandBuffer, pInfo);

    const uint32_t    n = pInfo->regionCount;
    VkImageResolve2   stack[BLIT_STACK_REGIONS];
    VkImageResolve2  *tmp = (n <= BLIT_STACK_REGIONS) ? stack :
                            (VkImageResolve2 *)malloc(sizeof(VkImageResolve2) * n);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdResolveImage2: OOM allocating %u regions", n);
        return;
    }
    memcpy(tmp, pInfo->pRegions, sizeof(VkImageResolve2) * n);

    for (uint32_t r = 0; r < n; r++)
    {
        if (src_upg)
            tmp[r].srcSubresource.baseArrayLayer = 1;
        if (dst_upg)
            tmp[r].dstSubresource.baseArrayLayer = 1;
    }

    VkResolveImageInfo2 info2 = *pInfo;
    info2.pRegions = tmp;
    sd->real.CmdResolveImage2(commandBuffer, &info2);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdCopyBufferToImage2  (Sync2 / VK 1.3 core)
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyBufferToImage2(
    VkCommandBuffer                commandBuffer,
    const VkCopyBufferToImageInfo2 *pInfo)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdCopyBufferToImage2)
        return;
    if (pInfo == NULL)
        return;
    if (pInfo->regionCount == 0 || pInfo->pRegions == NULL)
    {
        sd->real.CmdCopyBufferToImage2(commandBuffer, pInfo);
        return;
    }

    const bool dst_upg = stereo_image_is_layered2(sd, pInfo->dstImage);

    blit_log_buffer_image("CmdCopyBufferToImage2", pInfo->dstImage, dst_upg,
                          pInfo->regionCount, pInfo->pRegions);

    if (!dst_upg)
    {
        sd->real.CmdCopyBufferToImage2(commandBuffer, pInfo);
        return;
    }

    sd->real.CmdCopyBufferToImage2(commandBuffer, pInfo);

    const uint32_t        n = pInfo->regionCount;
    VkBufferImageCopy2    stack[BLIT_STACK_REGIONS];
    VkBufferImageCopy2   *tmp = (n <= BLIT_STACK_REGIONS) ? stack :
                                (VkBufferImageCopy2 *)malloc(sizeof(VkBufferImageCopy2) * n);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdCopyBufferToImage2: OOM allocating %u regions", n);
        return;
    }
    memcpy(tmp, pInfo->pRegions, sizeof(VkBufferImageCopy2) * n);

    for (uint32_t r = 0; r < n; r++)
    {
        tmp[r].imageSubresource.baseArrayLayer = 1;
    }

    VkCopyBufferToImageInfo2 info2 = *pInfo;
    info2.pRegions = tmp;
    sd->real.CmdCopyBufferToImage2(commandBuffer, &info2);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  stereo_CmdCopyImageToBuffer2  (Sync2 / VK 1.3 core)
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyImageToBuffer2(
    VkCommandBuffer                commandBuffer,
    const VkCopyImageToBufferInfo2 *pInfo)
{
    StereoDevice *sd = find_any_device_blit();

    if (!sd || !sd->real.CmdCopyImageToBuffer2)
        return;
    if (pInfo == NULL)
        return;
    if (pInfo->regionCount == 0 || pInfo->pRegions == NULL)
    {
        sd->real.CmdCopyImageToBuffer2(commandBuffer, pInfo);
        return;
    }

    const bool src_upg = stereo_image_is_layered2(sd, pInfo->srcImage);

    blit_log_buffer_image("CmdCopyImageToBuffer2", pInfo->srcImage, src_upg,
                          pInfo->regionCount, pInfo->pRegions);

    if (!src_upg)
    {
        sd->real.CmdCopyImageToBuffer2(commandBuffer, pInfo);
        return;
    }

    sd->real.CmdCopyImageToBuffer2(commandBuffer, pInfo);

    const uint32_t        n = pInfo->regionCount;
    VkBufferImageCopy2    stack[BLIT_STACK_REGIONS];
    VkBufferImageCopy2   *tmp = (n <= BLIT_STACK_REGIONS) ? stack :
                                (VkBufferImageCopy2 *)malloc(sizeof(VkBufferImageCopy2) * n);
    if (tmp == NULL)
    {
        STEREO_ERR("CmdCopyImageToBuffer2: OOM allocating %u regions", n);
        return;
    }
    memcpy(tmp, pInfo->pRegions, sizeof(VkBufferImageCopy2) * n);

    for (uint32_t r = 0; r < n; r++)
    {
        tmp[r].imageSubresource.baseArrayLayer = 1;
    }

    VkCopyImageToBufferInfo2 info2 = *pInfo;
    info2.pRegions = tmp;
    sd->real.CmdCopyImageToBuffer2(commandBuffer, &info2);
    if (tmp != stack)
        free(tmp);
}

/* ────────────────────────────────────────────────────────────────────── *
 *  KHR-promoted aliases — identical signatures; just forward to the
 *  core Sync2 wrappers.  Vulkan guarantees binary compatibility
 *  between the KHR extension variants and the promoted core variants.
 * ────────────────────────────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_CmdBlitImage2KHR(VkCommandBuffer cb, const VkBlitImageInfo2 *p)
{ stereo_CmdBlitImage2(cb, p); }

VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyImage2KHR(VkCommandBuffer cb, const VkCopyImageInfo2 *p)
{ stereo_CmdCopyImage2(cb, p); }

VKAPI_ATTR void VKAPI_CALL
stereo_CmdResolveImage2KHR(VkCommandBuffer cb, const VkResolveImageInfo2 *p)
{ stereo_CmdResolveImage2(cb, p); }

VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyBufferToImage2KHR(VkCommandBuffer cb, const VkCopyBufferToImageInfo2 *p)
{ stereo_CmdCopyBufferToImage2(cb, p); }

VKAPI_ATTR void VKAPI_CALL
stereo_CmdCopyImageToBuffer2KHR(VkCommandBuffer cb, const VkCopyImageToBufferInfo2 *p)
{ stereo_CmdCopyImageToBuffer2(cb, p); }
