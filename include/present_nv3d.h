/* present_nv3d.h — NVIDIA 3D Vision direct-mode presentation
 *
 * Only non-empty when the NV3D-Lib submodule is available.  For the common
 * case (missing submodule, e.g. Yuzu running SBS mode) we provide tiny
 * inline stubs that return "not supported" (false / VK_ERROR_INITIALIZATION_
 * FAILED) so the rest of the code paths still link without the external
 * import libs. */

#if !defined(VKS3D_HAVE_NV3DLIB) || (VKS3D_HAVE_NV3DLIB == 0)

#include "stereo_icd.h"  /* StereoDevice / StereoSwapchain / VkResult */

static inline bool
nv3d_init(
    StereoDevice* sd,
    uint32_t width,
    uint32_t height)
{
    (void)sd; (void)width; (void)height;
    return false;
}

static inline void
nv3d_destroy(
    StereoDevice* sd)
{
    (void)sd;
}

static inline VkResult
nv3d_present(
    StereoDevice* sd,
    StereoSwapchain* sc,
    VkQueue queue,
    uint32_t wait_sem_count,
    const VkSemaphore* wait_sems)
{
    (void)sd; (void)sc; (void)queue; (void)wait_sem_count; (void)wait_sems;
    return VK_ERROR_INITIALIZATION_FAILED;
}

#else   /* VKS3D_HAVE_NV3DLIB == 1 — real implementation lives in present_nv3d.cpp */

bool nv3d_init(
    StereoDevice* sd,
    uint32_t width,
    uint32_t height);

void nv3d_destroy(
    StereoDevice* sd);

VkResult nv3d_present(
    StereoDevice* sd,
    StereoSwapchain* sc,
    VkQueue queue,
    uint32_t wait_sem_count,
    const VkSemaphore* wait_sems);

#endif  /* VKS3D_HAVE_NV3DLIB */