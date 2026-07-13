/*
 * shader.c — SPIR-V stereo injection, deferred to pipeline creation
 *
 * Path A — pipeline has existing TCS+TES: patch TES with gl_ViewIndex.
 *
 * Path B — VS-based pipeline (no existing tessellation): patch VS directly
 *           with gl_ViewIndex. Works on any driver that properly implements
 *           VK_KHR_multiview (all current NVIDIA, AMD, Intel drivers).
 *           This replaces the TCS+TES injection approach which was a
 *           426.06-specific workaround and causes interface mismatch crashes
 *           on newer drivers due to strict PerVertex block validation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stereo_icd.h"
#include "tes_inject.h"
#include "spirv/unified1/spirv.h"

#define SpvExecVertex           0
#define SpvExecTessEval         2
#define SpvExecGeometry         3
#define SpvStorageOutput        3
#define SpvStorageInput         1
#define SPIRV_MAGIC             0x07230203u

/* ────────────────────────────────────────────────────────────────────────── */
/* Dynamic word buffer                                                       */
/* ────────────────────────────────────────────────────────────────────────── */

typedef struct
{
    uint32_t *w;
    size_t    n;
    size_t    cap;
} SpvBuf;

/* ── Matrix provenance helpers ───────────────────────────────────────────── */

static inline bool valid_id(const SpvMod *m, uint32_t id)
{
    return id < m->value_capacity;
}

static inline uint8_t matrix_value(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->value_from_matrix[id] : 0;
}

static inline void set_matrix_value(SpvMod *m, uint32_t id, uint8_t value)
{
    if (valid_id(m, id))
        m->value_from_matrix[id] = value;
}

static inline uint8_t matrix_ptr(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->is_matrix_ptr[id] : 0;
}

static inline void set_matrix_ptr(SpvMod *m, uint32_t id, uint8_t value)
{
    if (valid_id(m, id))
        m->is_matrix_ptr[id] = value;
}

static inline uint8_t matrix_type(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->is_matrix_type[id] : 0;
}

static inline void set_matrix_type(SpvMod *m, uint32_t id, uint8_t value)
{
    if (valid_id(m, id))
        m->is_matrix_type[id] = value;
}

static inline uint8_t matrix_or2(const SpvMod *m,
                                 uint32_t a,
                                 uint32_t b)
{
    return matrix_value(m, a) | matrix_value(m, b);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* SPIR-V module scan state                                                  */
/* ────────────────────────────────────────────────────────────────────────── */
typedef struct
{
    const uint32_t *words;
    size_t          count;
    uint32_t bound;
    bool is_patchable;
    bool has_mv_cap;
    /* Diagnostics */
    bool has_emit_vertex;
    bool has_viewindex_builtin;
    /* Execution model */
    int exec_model;
    /* Builtins */
    uint32_t pos_var;
    uint32_t pos_member_idx;
    uint32_t pos_ptr_type;
    bool     pos_is_block;
    uint32_t pos_block_type[8];
    uint32_t pos_block_count;
    uint32_t view_var;
    /* Common types */
    uint32_t ft;
    uint32_t v4t;
    uint32_t it;
    uint32_t bt;
    uint32_t ptr_out_v4;
    uint32_t ptr_in_int;
    /* Entry point */
    uint32_t entry_function;
    size_t entry_function_word;
    size_t fn_word;
    /* Function writing Position */
    uint32_t position_function;
    /* Geometry */
    uint32_t emit_count;
} SpvMod;

static uint64_t hash_spv(const uint32_t *data, size_t words);

static void do_scan(SpvMod *m, bool p2)
{
    const uint32_t *w=m->words;
    uint32_t current_function = 0;
    #define MAT(id)        matrix_value(m, (id))
    #define SETMAT(id,v)   set_matrix_value(m, (id), (v))
    #define PTR(id)        matrix_ptr(m, (id))
    #define SETPTR(id,v)   set_matrix_ptr(m, (id), (v))
    #define TYPE(id)       matrix_type(m, (id))
    #define SETTYPE(id,v)  set_matrix_type(m, (id), (v))
    for (size_t i=5;i<m->count;) {
        uint32_t op=w[i]&0xffff, wc=w[i]>>16;
        if (!wc||i+wc>m->count) break;
        if (!p2)
        {
        switch(op) {
            case SpvOpDot:
                m->dot_count++;
                break;
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
            case SpvOpPtrAccessChain:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETPTR(w[i + 2], PTR(w[i + 3]));
                }
                break;
            case SpvOpLoad:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(
                        w[i + 2],
                        MAT(w[i + 3]) || PTR(w[i + 3]));
                }
                break;
            case SpvOpCompositeExtract:
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                }
                break;
            case SpvOpVectorShuffle:
                if (wc >= 6 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                }
                break;
            case SpvOpCompositeConstruct:
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    for (uint32_t k = 3; k < wc; ++k)
                    {
                        if (w[i + k] < m->value_capacity &&
                            MAT(w[i + k]))
                        {
                            matrix = 1;
                            break;
                        }
                    }
                    SETMAT(w[i + 2], matrix);
                }
                break;
            case SpvOpCapability:
                if(wc>=2&&w[i+1]==SpvCapabilityMultiView) m->has_mv_cap=true;
                break;
            case SpvOpEntryPoint:
                if(wc>=3){
                    uint32_t e=w[i+1];
                    if(e==SpvExecVertex||e==SpvExecTessEval||e==SpvExecGeometry)
                    {
                        m->is_patchable=true;
                        m->exec_model=(int)e;
                        m->entry_function = w[i+2];
                    }}
                break;
            case SpvOpTypeFloat:
                if(wc==3&&w[i+2]==32) m->ft=w[i+1];
                break;
            case SpvOpTypeVector:
                if(wc==4&&w[i+2]==m->ft&&w[i+3]==4) m->v4t=w[i+1];
                break;
            case SpvOpTypeInt:
                if(wc==4&&w[i+2]==32) m->it=w[i+1];
                break;
            case SpvOpTypeMatrix:
                break;
            case SpvOpTypeStruct:
                if (wc >= 3)
                {
                    uint8_t matrix = 0;
                    for (uint32_t k = 2; k < wc; ++k)
                    {
                        if (w[i + k] < m->value_capacity &&
                            TYPE(w[i + k]))
                        {
                            matrix = 1;
                            break;
                        }
                    }
                    if (w[i + 1] < m->value_capacity)
                        SETTYPE(w[i + 1], matrix);
                }
                break;
            case SpvOpTypeArray:
                break;
            case SpvOpTypeRuntimeArray:
                break;
            case SpvOpTranspose:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                }
                break;
            case SpvOpMatrixTimesVector:
            case SpvOpMatrixTimesMatrix:
                m->has_matrix_ops = true;
                /* fall through */
            case SpvOpVectorTimesScalar:
            case SpvOpVectorTimesMatrix:
            case SpvOpMatrixTimesScalar:
                if (wc >= 5)
                {
                    if (w[i + 2] < m->value_capacity)
                    {
                        SETMAT(w[i + 2], 1);
                        // STEREO_LOG("MATRIX_MARK result=%u", w[i + 2]);
                    }
                    /*
                    else
                    {
                        STEREO_LOG(
                            "MATRIX_CAP_FAIL result=%u cap=%u",
                            w[i + 2],
                            m->value_capacity);
                    }
                    */
                }
                break;
            case SpvOpCopyObject:
            case SpvOpBitcast:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                }
                break;
            case SpvOpExtInst:
                if (wc >= 7 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    for (uint32_t k = 5; k < wc; ++k)
                    {
                        if (w[i + k] < m->value_capacity)
                            matrix |= MAT(w[i + k]);
                    }
                    SETMAT(w[i + 2], matrix);
                }
                break;
            case SpvOpFAdd:
            case SpvOpFSub:
            case SpvOpFMul:
            case SpvOpFDiv:
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity)
                {
                    SETMAT(
                        w[i + 2],
                        matrix_or2(m, w[i + 4], w[i + 5]));
                }
                break;
            case SpvOpSelect:
                if (wc >= 6 &&
                    w[i + 2] < m->value_capacity)
                {
                    SETMAT(
                        w[i + 2],
                        matrix_or2(m, w[i + 4], w[i + 5]));
                }
                break;
            case SpvOpFunctionCall:
                break;
            case SpvOpCompositeInsert:
                if (wc >= 6 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    if (w[i + 3] < m->value_capacity)
                        matrix |= MAT(w[i + 3]);
                    if (w[i + 4] < m->value_capacity)
                        matrix |= MAT(w[i + 4]);
                    SETMAT(w[i + 2], matrix);
                }
                break;
            case SpvOpTypePointer:
                if (wc >= 4)
                {
                if (TYPE(w[i + 3]))
                {
                SETPTR(w[i + 1], 1);
                }
                if (w[i + 2] == SpvStorageOutput &&
                m->v4t &&
                w[i + 3] == m->v4t)
                {
                m->ptr_out_v4 = w[i + 1];
                }
                if (w[i + 2] == SpvStorageInput &&
                m->it &&
                w[i + 3] == m->it)
                {
                m->ptr_in_int = w[i + 1];
                }
                }
                break;
            case SpvOpVariable:
                if (wc >= 4 &&
                    w[i + 1] < m->value_capacity &&
                    w[i + 2] < m->value_capacity &&
                    PTR(w[i + 1]))
                {
                SETPTR(w[i + 2], 1);
                }
                break;
            case SpvOpDecorate:
                if(wc>=4&&w[i+2]==SpvDecorationBuiltIn){
                    if(w[i+3]==SpvBuiltInPosition&&!m->pos_is_block)
                        m->pos_var=w[i+1];
                    if(w[i+3]==SpvBuiltInViewIndex) {
                        m->view_var = w[i+1];
                        m->has_viewindex_builtin = true;
                    }
                } break;
            case SpvOpMemberDecorate:
                if (wc >= 5 &&
                    w[i+3] == SpvDecorationBuiltIn &&
                    w[i+4] == SpvBuiltInPosition)
                {
                    if (m->pos_block_count < 8)
                        m->pos_block_type[m->pos_block_count++] = w[i+1];
                    m->pos_member_idx = w[i+2];
                    m->pos_is_block   = true;
                    m->pos_var        = 0;
                }
                break;
            case SpvOpFunction:
                if (!m->fn_word)
                    m->fn_word = i;
                if (wc >= 3)
                    current_function = w[i+2];
                break;
            case SpvOpFunctionEnd:
                current_function = 0;
                break;
            case SpvOpEmitVertex:
                m->emit_count++;
                m->has_emit_vertex = true;
                break;
            case SpvOpStore:
                if (wc >= 3 &&
                    w[i + 1] == m->pos_var)
                {
                    if (current_function &&
                        !m->position_function)
                    {
                        m->position_function = current_function;
                    }
                    uint32_t source = w[i + 2];
                    if (source >= m->value_capacity ||
                        !MAT(source))
                    {
                        m->has_direct_position_write = true;
                    }
                }
                break;
            }
        } else {
            if(op==SpvOpTypePointer && wc>=4 &&
               w[i+2]==SpvStorageOutput)
            {
                for(uint32_t k=0;k<m->pos_block_count;k++)
                {
                    if(w[i+3]==m->pos_block_type[k])
                    {
                        m->pos_ptr_type=w[i+1];
                        break;
                    }
                }
            }
            if(op==SpvOpVariable&&wc>=4&&w[i+3]==SpvStorageOutput)
            {
                if(m->pos_ptr_type &&
                   w[i+1]==m->pos_ptr_type)
                {
                    m->pos_var=w[i+2];
                }
            }
        }
        i+=wc;
    }
}

static void spv_scan(SpvMod *m)
{
    /* First pass: discover decorations/types. */
    do_scan(m, false);
    /*
     * Run again now that block Position info is known.
     * Some TES shaders declare OpTypePointer before
     * OpMemberDecorate(BuiltIn Position).
     */
    do_scan(m, false);
    if (m->pos_is_block)
        do_scan(m, true);
}

uint64_t hash_spv(const uint32_t *data, size_t words)
{
    uint64_t h = 1469598103934665603ULL; // FNV offset basis
    for (size_t i = 0; i < words; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

StereoPipelineInfo *
find_pipeline_info(
    StereoDevice *sd,
    VkPipeline pipeline)
{
    for (uint32_t i = 0; i < sd->pipeline_info_count; i++)
    {
        if (sd->pipeline_info[i].pipeline == pipeline)
            return &sd->pipeline_info[i];
    }

    return NULL;
}

VkPipeline
lookup_bound_pipeline(
    StereoDevice *sd,
    VkCommandBuffer cb)
{
    for (uint32_t i = 0; i < sd->cb_track_count; i++)
    {
        if (sd->cb_track[i].cb == cb)
            return sd->cb_track[i].pipeline;
    }

    return VK_NULL_HANDLE;
}

StereoDevice *
stereo_device_from_command_buffer(
    VkCommandBuffer cb)
{
    extern StereoDevice g_devices[];
    extern uint32_t g_device_count;

    for (uint32_t d = 0; d < g_device_count; d++)
    {
        StereoDevice *sd = &g_devices[d];

        for (uint32_t i = 0; i < sd->cb_track_count; i++)
        {
            if (sd->cb_track[i].cb == cb)
                return sd;
        }
    }

    return NULL;
}

void
remember_bound_pipeline(
    StereoDevice *sd,
    VkCommandBuffer cb,
    VkPipeline pipe)
{
    for (uint32_t i = 0; i < sd->cb_track_count; i++)
        {
        if (sd->cb_track[i].cb == cb)
        {
            sd->cb_track[i].pipeline = pipe;
            return;
            }
        }
        if (sd->cb_track_count >= MAX_CB_TRACK)
            return;
        uint32_t idx = sd->cb_track_count++;
        sd->cb_track[idx].cb = cb;
        sd->cb_track[idx].pipeline = pipe;
        sd->cb_track[idx].render_pass = VK_NULL_HANDLE;
        sd->cb_track[idx].framebuffer = VK_NULL_HANDLE;
        sd->cb_track[idx].subpass = 0;
}

void
remember_begin_renderpass(
    StereoDevice *sd,
    VkCommandBuffer cb,
    VkRenderPass rp,
    uint32_t subpass)
{
    for (uint32_t i = 0; i < sd->cb_track_count; i++)
    {
        if (sd->cb_track[i].cb == cb)
        {
            sd->cb_track[i].render_pass = rp;
            sd->cb_track[i].subpass = subpass;
            return;
        }
    }
    if (sd->cb_track_count >= MAX_CB_TRACK)
    return;
    uint32_t idx = sd->cb_track_count++;
    sd->cb_track[idx].cb = cb;
    sd->cb_track[idx].pipeline = VK_NULL_HANDLE;
    sd->cb_track[idx].render_pass = rp;
    sd->cb_track[idx].framebuffer = VK_NULL_HANDLE;
    sd->cb_track[idx].subpass = subpass;
}

VkRenderPass
lookup_bound_renderpass(
    StereoDevice *sd,
    VkCommandBuffer cb)
{
    for (uint32_t i = 0; i < sd->cb_track_count; i++)
    {
        if (sd->cb_track[i].cb == cb)
            return sd->cb_track[i].render_pass;
    }

    return VK_NULL_HANDLE;
}

VkFramebuffer
lookup_bound_framebuffer(
    StereoDevice *sd,
    VkCommandBuffer cb)
{
    for (uint32_t i = 0; i < sd->cb_track_count; i++)
    {
        if (sd->cb_track[i].cb == cb)
            return sd->cb_track[i].framebuffer;
    }

    return VK_NULL_HANDLE;
}

static StereoPipelineInfo *
add_pipeline_info(
    StereoDevice *sd)
{
    if (sd->pipeline_info_count >= sd->pipeline_info_capacity)
    {
        uint32_t new_cap =
            sd->pipeline_info_capacity ?
            sd->pipeline_info_capacity * 2 :
            128;
        StereoPipelineInfo *new_array =
            realloc(
            sd->pipeline_info,
            sizeof(*new_array) * new_cap);
        if (!new_array)
            return NULL;
        sd->pipeline_info = new_array;
        sd->pipeline_info_capacity = new_cap;
    }
    StereoPipelineInfo *info =
        &sd->pipeline_info[sd->pipeline_info_count++];
    memset(info, 0, sizeof(*info));
    return info;
}

/* ── Stereo offset injection body ────────────────────────────────────────── */
typedef struct {
    SpvMod  *m;
    bool     have_view;
    uint32_t uv4, uint_, bt;
    uint32_t cz;
    uint32_t cf0;
    uint32_t cl;
    uint32_t cr;
    uint32_t cc;
    uint32_t projection_mode;
    const StereoDebugCtx *dbg;
} BodyCtx;

typedef struct StereoDebugCtx {
    uint32_t pipeline_index;
    VkRenderPass render_pass;
    bool is_multiview;
    uint32_t stage;
    uint32_t vertex_binding_count;
    uint32_t is_quad;
} StereoDebugCtx;

static void emit_body(SpvBuf *out, const BodyCtx *c, uint32_t *nid)
{
    SpvMod *m = c->m;
    uint32_t ch = (*nid)++;
    uint32_t lp = (*nid)++;
    uint32_t pptr;
    if (m->pos_is_block)
    {
        uint32_t mid = (m->pos_member_idx == 0) ? c->cz : (*nid)++;
        if (m->pos_member_idx != 0)
        {
            uint32_t ci[] = {
                op_(SpvOpConstant, 4),
                m->it,
                mid,
                m->pos_member_idx
            };
            sb_push_n(out, ci, 4);
        }
        uint32_t a[] = {
            op_(SpvOpAccessChain, 5),
            c->uv4,
            ch,
            m->pos_var,
            mid
        };
        sb_push_n(out, a, 5);
        pptr = ch;
    }
    else
    {
        pptr = m->pos_var;
    }
    {
        uint32_t w[] = {
            op_(SpvOpLoad, 4),
            m->v4t,
            lp,
            pptr
        };
        sb_push_n(out, w, 4);
    }
    uint32_t lv = c->have_view ? (*nid)++ : 0;
    uint32_t isl = c->have_view ? (*nid)++ : 0;
    uint32_t sel = (*nid)++;
    uint32_t px = (*nid)++;
    uint32_t nx = (*nid)++;
    uint32_t np = (*nid)++;
    if (c->have_view && m->view_var && m->it && c->bt)
    {
        {
            uint32_t w[] = {
                op_(SpvOpLoad, 4),
                m->it,
                lv,
                m->view_var
            };
            sb_push_n(out, w, 4);
        }
        {
            uint32_t w[] = {
                op_(SpvOpIEqual, 5),
                c->bt,
                isl,
                lv,
                c->cz
            };
            sb_push_n(out, w, 5);
        }
        {
            uint32_t w[] = {
                op_(SpvOpSelect, 6),
                m->ft,
                sel,
                isl,
                c->cr,
                c->cl
            };
            sb_push_n(out, w, 6);
        }
    }
    else
    {
        sel = c->cl;
    }
    {
        uint32_t w[] = {
            op_(SpvOpCompositeExtract, 5),
            m->ft,
            px,
            lp,
            0u
        };
        sb_push_n(out, w, 5);
    }
    if (c->projection_mode == STEREO_PROJECTION_PARALLEL)
    {
        uint32_t w[] = {
            op_(SpvOpFAdd, 5),
            m->ft,
            nx,
            px,
            sel
        };
        sb_push_n(out, w, 5);
    }
    else
    {
        uint32_t pw = (*nid)++;
        uint32_t convmag = (*nid)++;
        uint32_t negconv = (*nid)++;
        uint32_t convsel = (*nid)++;
        uint32_t tmp = (*nid)++;
        {
            uint32_t w[] = {
                op_(SpvOpCompositeExtract, 5),
                m->ft,
                pw,
                lp,
                3u
            };
            sb_push_n(out, w, 5);
        }
        {
            uint32_t w[] = {
                op_(SpvOpFMul, 5),
                m->ft,
                convmag,
                pw,
                c->cc
            };
            sb_push_n(out, w, 5);
        }
        {
            uint32_t w[] = {
                op_(131, 5),
                m->ft,
                negconv,
                c->cf0,
                convmag
            };
            sb_push_n(out, w, 5);
        }
        {
            uint32_t w[] = {
                op_(SpvOpSelect, 6),
                m->ft,
                convsel,
                isl,
                convmag,
                negconv
            };
            sb_push_n(out, w, 6);
        }
        {
            uint32_t w[] = {
                op_(SpvOpFAdd, 5),
                m->ft,
                tmp,
                px,
                sel
            };
            sb_push_n(out, w, 5);
        }
        {
            uint32_t w[] = {
                op_(131, 5),
                m->ft,
                nx,
                tmp,
                convsel
            };
            sb_push_n(out, w, 5);
        }
    }
    {
        uint32_t w[] = {
            op_(SpvOpCompositeInsert, 6),
            m->v4t,
            np,
            nx,
            lp,
            0u
        };
        sb_push_n(out, w, 6);
    }
    {
        uint32_t w[] = {
            op_(SpvOpStore, 3),
            pptr,
            np
        };
        sb_push_n(out, w, 3);
    }
}

/* ── Public patcher ──────────────────────────────────────────────────────── */
bool spirv_patch_stereo_vertex(
    const StereoConfig *cfg,
    const uint32_t *in, size_t in_c,
    uint32_t **out, size_t *out_c,
    float lo, float ro,
    float conv,
    bool inj_vi,
    const StereoDebugCtx *dbg)
{
    const int projection_mode = cfg->projection;

    STEREO_LOG(
        "Projection=%s lo=%f ro=%f conv=%f",
        projection_mode == STEREO_PROJECTION_OFF_AXIS ?
            "off-axis" : "parallel",
        lo,
        ro,
        conv);
    if (!in||in_c<5||in[0]!=SPIRV_MAGIC) return false;

    SpvMod m={0};
    m.words=in;
    m.count=in_c;

    /* We need the bound before allocating the provenance table. */
    m.bound = m.words[3];
    m.value_capacity = m.bound + 64;
    m.value_from_matrix =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_matrix_type =
        calloc(m.value_capacity, sizeof(uint8_t));
    
    m.is_matrix_ptr =
        calloc(m.value_capacity, sizeof(uint8_t));
    if (!m.value_from_matrix)
        return false;

    spv_scan(&m);

    uint64_t spv_hash = hash_spv(m.words, m.count);
    {
        static int skip_list_init = 0;
        static char skip_list[1024];

        if (!skip_list_init)
        {
            const char *env = stereo_getenv("VKS3D_SKIP_SHADER_PATCHES");
            if (env)
            {
                strncpy(skip_list, env, sizeof(skip_list) - 1);
                skip_list[sizeof(skip_list) - 1] = '\0';
            }

            STEREO_LOG(
                "SKIP_SHADER_LIST=\"%s\"",
                skip_list);

            skip_list_init = 1;
        }

        if (skip_list[0])
        {
            char hashstr[17];
            snprintf(hashstr, sizeof(hashstr), "%016llx",
                (unsigned long long)spv_hash);

            if (strstr(skip_list, hashstr))
            {
                STEREO_LOG(
                    "SKIP_SHADER_PATCH hash=%s",
                    hashstr);
                free(m.value_from_matrix);
                free(m.is_matrix_type);
                free(m.is_matrix_ptr);
                return false;
            }
        }
    }
    if (cfg && cfg->mono_ui)
    {
        bool ui_candidate =
            (
                dbg &&
                (
                    dbg->is_quad ||
                    dbg->vertex_binding_count == 0
                )
            ) &&
            (m.dot_count <= 2) &&
            (m.has_direct_position_write) &&
            (!m.has_emit_vertex) &&
            (m.exec_model == SpvExecVertex);

        if (ui_candidate)
        {
            STEREO_LOG(
                "SCREENSPACE_SKIP hash=%016llx exec=%u pos=%u block=%u matrix=%u direct=%u emit=%u",
                (unsigned long long)spv_hash,
                (unsigned)m.exec_model,
                m.pos_var,
                m.pos_is_block,
                m.has_matrix_ops,
                m.has_direct_position_write,
                m.has_emit_vertex);

            free(m.value_from_matrix);
            free(m.is_matrix_type);
            free(m.is_matrix_ptr);
            return false;
        }
    }
    STEREO_LOG(
        "PATCH_MODULE hash=%016llx words=%zu module=%p",
        (unsigned long long)spv_hash,
        m.count,
        (const void *)m.words);
    STEREO_LOG(
        "PATCH_BEGIN hash=%016llx words=%zu bound=%u",
        (unsigned long long)spv_hash,
        m.count,
        m.bound);
    STEREO_LOG(
        "SCAN_CLASS hash=%016llx exec=%u patchable=%d pos=%u posBlock=%d posMember=%u view=%u emit=%u matrix=%d directPos=%d dots=%u mvbuiltin=%d",
        (unsigned long long)spv_hash,
        m.exec_model,
        m.is_patchable,
        m.pos_var,
        m.pos_is_block,
        m.pos_member_idx,
        m.view_var,
        m.emit_count,
        m.has_matrix_ops,
        m.has_direct_position_write,
        m.dot_count,
        m.has_viewindex_builtin);
    if (dbg)
    {
        STEREO_LOG(
            "PATCH_CTX hash=%016llx pipe=%u stage=%u rp=%p mv=%d",
            (unsigned long long)spv_hash,
            dbg->pipeline_index,
            dbg->stage,
            (void*)dbg->render_pass,
            dbg->is_multiview);
    }
    STEREO_LOG(
    "PATCHABLE hash=%016llx words=%zu exec=%u matrix=%d direct=%d dots=%u block=%d emits=%u pos=%u view=%u",
        (unsigned long long)spv_hash,
        m.count,
        m.exec_model,
        m.has_matrix_ops,
        m.has_direct_position_write,
        m.dot_count,
        m.pos_is_block,
        m.emit_count,
        m.pos_var,
        m.view_var);
    if (dbg) {
        STEREO_LOG(
            "PATCH_CTX pipe=%u stage=%u renderPass=%p multiview=%d",
            dbg->pipeline_index,
            dbg->stage,
            (void*)dbg->render_pass,
            dbg->is_multiview);
    
        if (!dbg->is_multiview) {
            STEREO_LOG(
                "PATCH_SKIP non-multiview render pass");
            free(m.value_from_matrix);
            free(m.is_matrix_type);
            free(m.is_matrix_ptr);
            return false;
        }
    }

    if (m.exec_model == SpvExecVertex)
    {
        STEREO_LOG(
            "SPIRV classify: vertex shader matrix_ops=%d",
            m.has_matrix_ops);
    }

    /* HUD/text/fullscreen shaders often write clip-space coordinates
     * directly and contain no matrix math. Stereoizing them pushes
     * them in front of the screen and causes excessive negative
     * parallax.
     *
     * Examples:
     *   gl_Position = vec4(pos.xy, 0.0, 1.0);
     *
     * Leave these monoscopic at screen depth.
     */
    if (m.exec_model == SpvExecVertex)
    {
        if (!m.pos_var)
        {
            STEREO_LOG("Skipping stereo patch: no gl_Position detected");
            free(m.value_from_matrix);
            free(m.is_matrix_type);
            free(m.is_matrix_ptr);
            return false;
        }
    
        /* Only reject truly screen-aligned fullscreen quads */
        if (m.pos_is_block && !m.has_matrix_ops)
        {
            STEREO_LOG("Skipping stereo patch: confirmed screen-space (pos block)");
            free(m.value_from_matrix);
            free(m.is_matrix_type);
            free(m.is_matrix_ptr);
            return false;
        }
    
        /* IMPORTANT: DO NOT rely on matrix_ops for DXVK */
    }

    if (!m.is_patchable)
    {
        free(m.value_from_matrix);
        free(m.is_matrix_type);
        free(m.is_matrix_ptr);
        return false;
    }

    /* Avoid stereoizing helper/fullscreen shaders that directly
     * write clip-space positions. These are responsible for the
     * duplicated shadow/composite artifacts seen in deferredshadows.
     */
/*     if (!m.has_matrix_ops && m.exec_model != SpvExecutionModelVertex) {*/
/*         STEREO_LOG("Skipping stereo patch: no matrix operations detected");*/
/*         return false;*/
/*     }*/

    bool is_gs = (m.exec_model == SpvExecGeometry);

    uint32_t nid=m.bound;
    uint32_t id_ptr_v4=nid++, id_ptr_int=nid++;
    uint32_t id_new_it=0;
    if (!m.it && inj_vi && !m.view_var) { id_new_it=nid++; m.it=id_new_it; }

    bool     will_inj_vi = inj_vi && !m.view_var && m.it;
    uint32_t id_inj_view = will_inj_vi ? nid++ : 0;
    bool     have_view   = (m.view_var || will_inj_vi);
    uint32_t id_new_bt=0;
    if (!m.bt && have_view && m.it) id_new_bt=nid++;

    uint32_t id_cz=nid++,
         id_cf0=nid++,
         id_cl=nid++,
         id_cr=nid++,
         id_cc=nid++;
    uint32_t uv4  = m.ptr_out_v4 ? m.ptr_out_v4 : id_ptr_v4;
    uint32_t uint_= m.ptr_in_int  ? m.ptr_in_int  : id_ptr_int;
    uint32_t bt   = m.bt          ? m.bt          : id_new_bt;

    SpvBuf te;
    if (!sb_init(&te,96))
    {
        free(m.value_from_matrix);
        free(m.is_matrix_type);
        free(m.is_matrix_ptr);
        return false;
    }
    if (id_new_it) { uint32_t w[]={op_(SpvOpTypeInt,4),id_new_it,32,1}; sb_push_n(&te,w,4); }
    if (!m.ptr_out_v4) {
        uint32_t w[]={op_(SpvOpTypePointer,4),id_ptr_v4,SpvStorageOutput,m.v4t};
        sb_push_n(&te,w,4); }
    if (m.it && !m.ptr_in_int) {
        uint32_t w[]={op_(SpvOpTypePointer,4),id_ptr_int,SpvStorageInput,m.it};
        sb_push_n(&te,w,4); m.ptr_in_int=id_ptr_int; uint_=id_ptr_int; }
    if (id_new_bt) { uint32_t w[]={op_(SpvOpTypeBool,2),id_new_bt}; sb_push_n(&te,w,2); }
    if (m.it) { uint32_t w[]={op_(SpvOpConstant,4),m.it,id_cz,0}; sb_push_n(&te,w,4); }
    STEREO_LOG(
        "[SPIRV] lo=%f ro=%f conv=%f projection=%d",
        lo,
        ro,
        conv,
        projection_mode);
    {
        uint32_t w[4]={op_(SpvOpConstant,4),m.ft,id_cf0,0};
        float z=0.0f;
        memcpy(&w[3],&z,4);
        sb_push_n(&te,w,4);
    }
    { uint32_t w[4]={op_(SpvOpConstant,4),m.ft,id_cl,0}; memcpy(&w[3],&lo,4); sb_push_n(&te,w,4); }
    { uint32_t w[4]={op_(SpvOpConstant,4),m.ft,id_cr,0}; memcpy(&w[3],&ro,4); sb_push_n(&te,w,4); }
    { uint32_t w[4]={op_(SpvOpConstant,4),m.ft,id_cc,0};
      memcpy(&w[3],&conv,4);
      sb_push_n(&te,w,4); }
    if (will_inj_vi) {
        { uint32_t d[]={op_(SpvOpDecorate,4),id_inj_view,SpvDecorationBuiltIn,SpvBuiltInViewIndex};
          sb_push_n(&te,d,4); }
        { uint32_t v[]={op_(SpvOpVariable,4),uint_,id_inj_view,SpvStorageInput};
          sb_push_n(&te,v,4); }
        m.view_var=id_inj_view;
    }

    BodyCtx bc = {
        &m,
        have_view,
        uv4,
        uint_,
        bt,
        id_cz,
        id_cf0,
        id_cl,
        id_cr,
        id_cc,
        projection_mode,
        lo,
        ro,
        cfg ? cfg->flip_eyes : 0,
        dbg
    };
    STEREO_LOG(
        "PATCH_BODY hash=%016llx lo=%f ro=%f conv=%f have_view=%d pos=%u",
        (unsigned long long)spv_hash,
        lo,
        ro,
        conv,
        have_view,
        m.pos_var);
    STEREO_LOG(
        "[SPIRV] build BodyCtx lo=%f ro=%f conv=%f proj=%d",
        lo,
        ro,
        conv,
        projection_mode);
    size_t ins_t=0, ins_b=0;
    bool in_entry_function=false;
    for (size_t i=5;i<in_c;) {
        uint32_t opx=in[i]&0xffff, wcx=in[i]>>16;
        if (!wcx||i+wcx>in_c) break;
        if (opx==SpvOpFunction) {
            in_entry_function =
                (wcx >= 4 &&
                 in[i+2] ==
                 (m.position_function ?
                     m.position_function :
                     m.entry_function));
            STEREO_LOG(
                "FUNCTION_START offset=%zu result=%u entry=%d",
                i,
                in[i+2],
                in_entry_function);
            if (in_entry_function)
                ins_t=i;
        }
        /* Always inject immediately before the final OpReturn.
         * Some shaders continue modifying gl_Position after its
         * last apparent OpStore via helper logic or additional
         * stores. Making the stereo adjustment the final operation
         * guarantees it survives.
         */
        if (in_entry_function && opx==SpvOpReturn) {
            STEREO_LOG(
                "RETURN offset=%zu",
                i);
            ins_b=i;
        }
    
        if (in_entry_function &&
            opx == SpvOpFunctionEnd)
        {
            STEREO_LOG(
                "FUNCTION_END offset=%zu return=%zu",
                i,
                ins_b);
            /* Finished scanning the entry function. */
            break;
        }
    
        i+=wcx;
    }
    STEREO_LOG(
        "INSERT_POINTS entry=%u position=%u function=%zu return=%zu",
        m.entry_function,
        m.position_function,
        ins_t,
        ins_b);
    if (!ins_t) { sb_free(&te); free(m.value_from_matrix); free(m.is_matrix_type); free(m.is_matrix_ptr); return false; }
    if (!is_gs && !ins_b) { sb_free(&te); free(m.value_from_matrix); free(m.is_matrix_type); free(m.is_matrix_ptr); return false; }
    if (!is_gs && (!ins_b || ins_b < ins_t)) { sb_free(&te); free(m.value_from_matrix); free(m.is_matrix_type); free(m.is_matrix_ptr); return false; }

    bool need_mv_cap = id_inj_view && !m.has_mv_cap;
    bool mv_done=false, te_done=false, body_done=false;

    SpvBuf ob;
    if (!sb_init(&ob, in_c + te.n + 64)) { sb_free(&te); free(m.value_from_matrix); free(m.is_matrix_type); free(m.is_matrix_ptr); return false; }
    sb_push_n(&ob, in, 5);

    for (size_t i=5;i<in_c;) {
        if (!mv_done && need_mv_cap) {
            uint32_t c[]={op_(SpvOpCapability,2),SpvCapabilityMultiView};
            sb_push_n(&ob,c,2); mv_done=true; }
        if (!te_done && i==ins_t) { sb_push_n(&ob,te.w,te.n); te_done=true; }

        uint32_t opx=in[i]&0xffff, wcx=in[i]>>16;
        if (!wcx||i+wcx>in_c) break;

        if (id_inj_view && opx==SpvOpEntryPoint && wcx>=4 &&
            (in[i+1]==SpvExecVertex||in[i+1]==SpvExecGeometry||
             in[i+1]==SpvExecTessEval)) {
                bool is_target_entry =
                    (wcx >= 3 &&
                     in[i+2] == m.entry_function);
                if (id_inj_view && is_target_entry)
            {
                sb_push(&ob, ((wcx+1)<<16)|SpvOpEntryPoint);
                sb_push_n(&ob, &in[i+1], wcx-1);
                sb_push(&ob, id_inj_view);
            }
            else
            {
                sb_push_n(&ob, &in[i], wcx);
            }
            i+=wcx; continue;
        }

        if (is_gs && opx==SpvOpEmitVertex) emit_body(&ob, &bc, &nid);
        if (!is_gs && !body_done && i==ins_b) { emit_body(&ob, &bc, &nid); body_done=true; }

        sb_push_n(&ob, &in[i], wcx);
        i+=wcx;
    }
    if (!te_done) sb_push_n(&ob,te.w,te.n);
    sb_free(&te);
    ob.w[3]=nid;
    *out=ob.w; *out_c=ob.n;
    STEREO_LOG("Patched: model=%d  %zu->%zu words  bound=%u  vi=%d",
               m.exec_model, in_c, ob.n, nid, (int)(id_inj_view!=0));
    free(m.value_from_matrix);
    free(m.is_matrix_type);
    free(m.is_matrix_ptr);
    return true;
}

void spirv_patched_free(uint32_t *w) { free(w); }

/* ══════════════════════════════════════════════════════════════════════════
 * FS SPIR-V patcher — sampler2D → sampler2DArray + gl_ViewIndex layer
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Called for FULL-SCREEN QUAD pipelines (vertexBindingDescriptionCount == 0)
 * in multiview render passes.  All VkImage 2D attachments are upgraded to
 * arrayLayers=2 by stereo_CreateImage, so ALL sampler2D in these shaders
 * (G-buffer, shadow map, lighting output) reference 2D array images.
 * We patch ALL OpTypeImage Dim=2D Arrayed=0 → Arrayed=1 and extend the
 * sampling coordinate from vec2(u,v) to vec3(u,v,gl_ViewIndex).
 *
 * Geometry pipelines (has vertex input) use the existing VS gl_ViewIndex
 * patch instead — those shaders sample material textures (not upgraded) and
 * must NOT have their sampler types changed.
 */

#define FS_MAX_IMG        64
#define FS_MAX_SI         64
#define FS_MAX_LOADS     512
#define FS_MAX_PARAMS    256
#define FS_MAX_CALLS     256
#define FS_MAX_FUNCTIONS  64

typedef struct {
    uint32_t img_ids[FS_MAX_IMG];       uint32_t n_img;
    uint32_t img_patchable[FS_MAX_IMG];
    uint32_t img_depth[FS_MAX_IMG];
    uint32_t img_arrayed[FS_MAX_IMG];

    uint32_t si_ids[FS_MAX_SI];         uint32_t n_si;

    uint32_t load_ids[FS_MAX_LOADS];
    uint32_t load_vars[FS_MAX_LOADS];
    uint32_t load_bindings[FS_MAX_LOADS];
    uint32_t n_load;
    /* Function parameter descriptor ownership */
    uint32_t param_ids[FS_MAX_PARAMS];
    uint32_t param_vars[FS_MAX_PARAMS];
    uint32_t param_functions[FS_MAX_PARAMS];
    uint32_t n_param;
    /* Function call argument bindings */
    uint32_t call_functions[FS_MAX_CALLS];
    uint32_t call_params[FS_MAX_CALLS];
    uint32_t call_args[FS_MAX_CALLS];
    uint32_t n_call;
    /* Descriptor variable tracking */
#define FS_MAX_VARS 128
    uint32_t var_ids[FS_MAX_VARS];
    uint32_t var_types[FS_MAX_VARS];
    uint32_t var_set[FS_MAX_VARS];

    /* Decorations can legally appear before OpVariable. */
    uint32_t dec_target[FS_MAX_VARS];
    uint32_t dec_binding[FS_MAX_VARS];
    uint32_t dec_set[FS_MAX_VARS];
    uint32_t n_dec;

    uint32_t var_binding[FS_MAX_VARS];
    uint32_t var_location[FS_MAX_VARS];
    uint32_t n_var;

    uint32_t float_id;
    uint32_t int_id;
    uint32_t v3float_id;
    uint32_t ptr_int_in_id;
    uint32_t vi_var_id;
    bool     has_mv_cap;
    size_t   ep_word;
    size_t   fn_word;
    
    /* Current function tracking */
    uint32_t current_function_id;
    uint32_t current_param_index;
    uint32_t function_ids[FS_MAX_FUNCTIONS];
    uint32_t function_param_start[FS_MAX_FUNCTIONS];
    uint32_t n_function;
} FsScan;

static bool fs_id_in(const uint32_t *arr, uint32_t n, uint32_t id)
{
    for (uint32_t i = 0; i < n; i++) if (arr[i] == id) return true;
    return false;
}

static int fs_var_index(const FsScan *s, uint32_t id)
{
    for (uint32_t i = 0; i < s->n_var; i++)
    {
        if (s->var_ids[i] == id)
        {
            STEREO_LOG(
                "FS_VAR_LOOKUP id=%u index=%u set=%u binding=%u type=%u",
                id,
                i,
                s->var_set[i],
                s->var_binding[i],
                s->var_types[i]);
            return (int)i;
        }
    }
    STEREO_LOG(
        "FS_VAR_LOOKUP_MISS id=%u",
        id);
    return -1;
}

static uint32_t fs_resolve_parameter_owner(
    const FsScan *s,
    uint32_t id)
{
    for (uint32_t i = 0; i < s->n_call; ++i)
    {
        if (s->call_params[i] == id)
        {
            STEREO_LOG(
                "FS_PARAM_OWNER_RESOLVE param=%u owner=%u",
                id,
                s->call_args[i]);
            return s->call_args[i];
        }
    }
    return id;
}

static int fs_dec_index(FsScan *s, uint32_t target)
{
    for (uint32_t i = 0; i < s->n_dec; ++i)
    {
        if (s->dec_target[i] == target)
            return (int)i;
    }
    return -1;
}

static bool fs_binding_is_stereo_attachment(const FsScan *s, uint32_t var)
{
    int vi = fs_var_index(s, var);
    if (vi < 0)
    {
        STEREO_LOG(
            "FS_BINDING_TEST_MISS var=%u reason=no_var",
            var);
        return false;
    }
    uint32_t binding = s->var_binding[vi];
    uint32_t set     = s->var_set[vi];
    uint32_t type    = s->var_types[vi];
    STEREO_LOG(
        "FS_BINDING_TEST var=%u vi=%d set=%u binding=%u type=%u",
        var,
        vi,
        set,
        binding,
        s->var_types[vi]);
    /*
     * Deferred framebuffer attachments become stereo arrays.
     *
     * binding 0 = position/depth
     * binding 1 = normal
     * binding 2 = albedo
     * binding 3 = specular
     *
     * Later bindings are post-processing/noise/material resources
     * and remain mono.
     * binding 4 = SSAO/deferred intermediate (also stereo)
     *
     * Later bindings may be post-processing/noise/material resources
     * and remain mono.
     */
    bool result = (binding <= 4);
    STEREO_LOG(
        "FS_BINDING_CLASSIFY var=%u set=%u binding=%u type=%u stereo=%u",
        var,
        set,
        binding,
        type,
        result);
    STEREO_LOG(
        "FS_BINDING_TEST var=%u vi=%d set=%u binding=%u type=%u",
        var,
        vi,
        set,
        binding,
        type);
    return result;
}

static const char *spv_op_name(uint32_t op)
{
    switch (op)
    {
    case SpvOpCopyObject:
        return "OpCopyObject";
    case SpvOpImageSampleImplicitLod:
        return "OpImageSampleImplicitLod";
    case SpvOpImageSampleExplicitLod:
        return "OpImageSampleExplicitLod";
    case SpvOpImageSampleDrefImplicitLod:
        return "OpImageSampleDrefImplicitLod";
    case SpvOpImageSampleDrefExplicitLod:
        return "OpImageSampleDrefExplicitLod";
    case SpvOpImageFetch:
        return "OpImageFetch";
    case SpvOpImageRead:
        return "OpImageRead";
    case SpvOpImageWrite:
        return "OpImageWrite";
    case SpvOpImage:
        return "OpImage";
    case SpvOpImageGather:
        return "OpImageGather";
    case SpvOpImageDrefGather:
        return "OpImageDrefGather";
    case SpvOpImageSparseSampleImplicitLod:
        return "OpImageSparseSampleImplicitLod";
    case SpvOpImageSparseSampleExplicitLod:
        return "OpImageSparseSampleExplicitLod";
    case SpvOpImageSparseFetch:
        return "OpImageSparseFetch";
    case SpvOpImageSparseRead:
        return "OpImageSparseRead";
    case SpvOpImageSparseTexelsResident:
        return "OpImageSparseTexelsResident";
    case SpvOpFunctionParameter:
        return "OpFunctionParameter";
    case SpvOpVariable:
        return "OpVariable";
    case SpvOpSampledImage:
        return "OpSampledImage";
    case SpvOpLoad:
        return "OpLoad";
    default:
        return "Unknown";
    }
}

static bool fs_is_image_related_type(FsScan *s, uint32_t type)
{
    if (fs_id_in(s->img_ids, s->n_img, type))
        return true;

    if (fs_id_in(s->si_ids, s->n_si, type))
        return true;

    return false;
}

static void fs_prescan(FsScan *s, const uint32_t *w, size_t c)
{
    memset(s, 0, sizeof(*s));
    bool in_func = false;
    for (size_t i = 5; i < c; ) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (!wc || i + wc > c) break;
        /* Trace the instruction that defines id 15. */
        if (wc >= 3 && w[i+2] == 15)
        {
            STEREO_LOG(
                "FS_DEFINE15 op=%s(%u) type=%u result=%u wordcount=%u",
                spv_op_name(op),
                op,
                w[i+1],
                w[i+2],
                wc);
            STEREO_LOG(
                "FS_DEFINE15_RAW opcode=%u word0=0x%08x",
                op,
                w[i]);
            for (uint32_t k = 0; k < wc; ++k)
            {
                STEREO_LOG(
                    "FS_DEFINE15_WORD[%u]=%u",
                    k,
                    w[i+k]);
            }
            for (uint32_t k = 3; k < wc; ++k)
            {
                STEREO_LOG(
                    "FS_DEFINE15 operand[%u]=%u",
                    k,
                    w[i+k]);
            }
        }
        switch (op) {
        case SpvOpCapability:
            if (wc >= 2 && w[i+1] == 4439) s->has_mv_cap = true;
            break;
        case SpvOpEntryPoint:
            if (!s->ep_word) s->ep_word = i;
            break;
        case SpvOpTypeFloat:  /* OpTypeFloat 32 */
            if (wc >= 3 && w[i+2] == 32) s->float_id = w[i+1];
            break;
        case SpvOpTypeInt:  /* OpTypeInt 32 */
            if (wc >= 3 && w[i+2] == 32) s->int_id = w[i+1];
            break;
        case SpvOpTypeVector:  /* OpTypeVector float 3 */
            if (wc >= 4 && s->float_id && w[i+2] == s->float_id && w[i+3] == 3)
                s->v3float_id = w[i+1];
            break;
        case SpvOpTypeImage:  /* OpTypeImage */
        {
            if (wc >= 9)
            {
                STEREO_LOG(
                    "FS_IMAGE_TYPE id=%u sampledType=%u dim=%u depth=%u arrayed=%u ms=%u sampled=%u format=%u",
                    w[i+1],
                    w[i+2],
                    w[i+3],
                    w[i+4],
                    w[i+5],
                    w[i+6],
                    w[i+7],
                    w[i+8]);
                STEREO_LOG(
                    "FS image candidate id=%u type=%u dim=%u arrayed=%u",
                    w[i+1],
                    w[i+2],
                    w[i+3],
                    w[i+5]);
                /* Existing path */
                if (w[i+3] == 1 && w[i+5] == 0 && s->n_img < FS_MAX_IMG)
                {
                    /*
                     * Record the image type only.
                     *
                     * Do NOT patch here. At this point we only know:
                     *
                     *   Dim=2D
                     *   Arrayed=0
                     *
                     * This includes:
                     *   - G-buffer attachments
                     *   - SSAO depth textures
                     *   - normal textures
                     *   - noise textures
                     *   - material textures
                     *
                     * The descriptor binding must decide later.
                     */
                    STEREO_LOG(
                        "FS image candidate type=%u awaiting descriptor correlation",
                        w[i+1]);
                    s->img_ids[s->n_img++] = w[i+1];
                }
                else
                {
                    STEREO_LOG(
                        "FS rejected image type %u dim=%u sampled=%u",
                        w[i+1],
                        w[i+3],
                        w[i+7]);
                }
            }
        }
        break;
        case SpvOpTypeSampledImage:  /* OpTypeSampledImage: [1]=id [2]=image_type */
            if (wc >= 3 && fs_id_in(s->img_ids, s->n_img, w[i+2]) && s->n_si < FS_MAX_SI)
                s->si_ids[s->n_si++] = w[i+1];
            break;
        case SpvOpTypePointer:  /* OpTypePointer Input int → ptr_int_in */
            if (wc >= 4 && w[i+2] == 1 && s->int_id && w[i+3] == s->int_id)
                s->ptr_int_in_id = w[i+1];
            break;
        case SpvOpVariable:  /* OpVariable */
        {
            if (wc >= 4 && s->n_var < FS_MAX_VARS)
            {
                if (w[i+3] == 2) /* Uniform storage */
                {
                    STEREO_LOG(
                        "FS_UNIFORM_BUFFER var=%u type=%u",
                        w[i+2],
                        w[i+1]);
                }
                uint32_t idx = s->n_var++;
                s->var_ids[idx] = w[i+2];
                s->var_types[idx] = w[i+1];
                s->var_set[idx] = 0xffffffffu;
                s->var_binding[idx] = 0xffffffffu;
                s->var_location[idx] = 0xffffffffu;
                STEREO_LOG(
                    "FS_VAR_ADD id=%u idx=%u type=%u storage=%u set=%u binding=%u",
                    w[i+2],
                    idx,
                    w[i+1],
                    w[i+3],
                    s->var_set[idx],
                    s->var_binding[idx]);
                STEREO_LOG(
                    "FS_VAR_DECLARE id=%u type=%u storage=%u",
                    w[i+2],
                    w[i+1],
                    w[i+3]);
                /* Apply any cached descriptor decorations. */
                for (uint32_t d = 0; d < s->n_dec; ++d)
                {
                    if (s->dec_target[d] == w[i+2] &&
                        s->dec_binding[d] != 0xffffffffu &&
                        s->dec_set[d] != 0xffffffffu)
                    {
                        s->var_binding[idx] = s->dec_binding[d];
                        s->var_set[idx]     = s->dec_set[d];
                        STEREO_LOG(
                            "FS_DECORATION_APPLY var=%u set=%u binding=%u",
                            w[i+2],
                            s->dec_set[d],
                            s->dec_binding[d]);
                        STEREO_LOG(
                            "FS_DECORATION_STATE var=%u finalSet=%u finalBinding=%u",
                            w[i+2],
                            s->var_set[idx],
                            s->var_binding[idx]);
                        break;
                    }
                }
                STEREO_LOG(
                    "FS var declare id=%u type=%u storage=%u",
                    w[i+2],
                    w[i+1],
                    w[i+3]);
            }
        }
        break;
        case SpvOpDecorate:
            if (wc >= 4) {
                STEREO_LOG(
                    "FS_DECORATE target=%u decoration=%u literal=%u",
                    w[i+1],
                    w[i+2],
                    w[i+3]);
                if (w[i+2] == 30)
                {
                    int li = fs_var_index(s, w[i+1]);
                    if (li >= 0)
                        s->var_location[li] = w[i+3];
                }
                /* BuiltIn ViewIndex */
                if (w[i+2] == 11 && w[i+3] == 4440)
                    s->vi_var_id = w[i+1];
                /* Descriptor binding */
                if (w[i+2] == 33) {
                    int di = fs_dec_index(s, w[i+1]);
                    if (di < 0 && s->n_dec < FS_MAX_VARS)
                    {
                        di = s->n_dec++;
                        s->dec_target[di]  = w[i+1];
                        s->dec_binding[di] = 0xffffffffu;
                        s->dec_set[di]     = 0xffffffffu;
                    }
                    if (di >= 0)
                    {
                        s->dec_binding[di] = w[i+3];
                        STEREO_LOG(
                            "FS_BIND_CACHE target=%u binding=%u",
                            w[i+1],
                            w[i+3]);
                    }
                }
                /* Descriptor set */
                if (w[i+2] == 34) {
                    int di = fs_dec_index(s, w[i+1]);
                
                    if (di < 0 && s->n_dec < FS_MAX_VARS)
                    {
                        di = s->n_dec++;
                        s->dec_target[di]  = w[i+1];
                        s->dec_binding[di] = 0xffffffffu;
                        s->dec_set[di]     = 0xffffffffu;
                    }
                
                    if (di >= 0)
                    {
                        s->dec_set[di] = w[i+3];
                    }
                }
            }
            break;
        case SpvOpFunction:
            if (!s->fn_word)
                s->fn_word = i;
        
            in_func = true;
            s->current_function_id = w[i+2];
            s->current_param_index = 0;
        
            if (s->n_function < FS_MAX_FUNCTIONS)
            {
                s->function_ids[s->n_function] = w[i+2];
                s->function_param_start[s->n_function] = s->n_param;
                s->n_function++;
            }
        
            STEREO_LOG(
                "FS_FUNCTION_BEGIN id=%u",
                s->current_function_id);
        break;
        case SpvOpFunctionParameter:
            if (wc >= 3 &&
                s->n_param < FS_MAX_PARAMS)
            {
                STEREO_LOG(
                    "FS_PARAM_REGISTER function=%u param=%u type=%u",
                    s->current_function_id,
                    w[i+2],
                    w[i+1]);
                STEREO_LOG(
                    "FS_FUNCTION_PARAM function=%u index=%u id=%u type=%u",
                    s->current_function_id,
                    s->current_param_index,
                    w[i+2],
                    w[i+1]);
                s->param_ids[s->n_param] = w[i+2];
                s->param_vars[s->n_param] = 0;
                s->param_functions[s->n_param] = s->current_function_id;
                s->n_param++;
            }
            s->current_param_index++;
        break;
        default:
            if (in_func) {
                if (op == SpvOpFunctionCall && wc >= 4)
                {
                    STEREO_LOG(
                        "FS_CALL_TRACE result=%u function=%u wc=%u",
                        w[i+2],
                        w[i+3],
                        wc);
                    for (uint32_t a = 4; a < wc; ++a)
                    {
                        STEREO_LOG(
                            "FS_CALL_ARG_TRACE index=%u value=%u",
                            a - 4,
                            w[i+a]);
                    }
                    STEREO_LOG(
                        "FS_FUNCTION_CALL result=%u function=%u argc=%u",
                        w[i+2],
                        w[i+3],
                        wc - 4);
                    STEREO_LOG(
                        "FS_FUNCTION_CALL_FIRST_ARG function=%u arg0=%u",
                        w[i+3],
                        wc > 4 ? w[i+4] : 0);
                    for (uint32_t k = 4; k < wc; ++k)
                    {
                        STEREO_LOG(
                            "FS_FUNCTION_ARG[%u]=%u",
                            k - 4,
                            w[i+k]);
                        uint32_t arg_index = k - 4;
                         /*
                          * Store call arguments immediately.
                          * Function parameters may appear later in the module.
                          */
                         if (s->n_call < FS_MAX_CALLS)
                         {
                             s->call_functions[s->n_call] = w[i+3];
                             s->call_params[s->n_call] = arg_index;
                             s->call_args[s->n_call] = w[i+k];
                             s->n_call++;
                             STEREO_LOG(
                                 "FS_CALL_STORE function=%u argIndex=%u value=%u total=%u",
                                 w[i+3],
                                 arg_index,
                                 w[i+k],
                                 s->n_call);
                        }
                    }
                }
                if ((op == SpvOpImageSampleImplicitLod ||
                     op == SpvOpImageSampleExplicitLod ||
                     op == SpvOpImageSampleDrefImplicitLod ||
                     op == SpvOpImageSampleDrefExplicitLod ||
                     op == SpvOpImageFetch ||
                     op == SpvOpImageRead ||
                     op == SpvOpImageWrite) && wc >= 5)
                {
                    STEREO_LOG(
                        "FS_IMAGE_OP op=%s(%u) type=%u result=%u image=%u coord=%u",
                        spv_op_name(op),
                        op,
                        w[i+1],
                        w[i+2],
                        w[i+3],
                        w[i+4]);
                    STEREO_LOG(
                        "FS_SAMPLE_COORD result=%u coord=%u",
                        w[i+2],
                        w[i+4]);
                    uint32_t descriptor_var = 0;
                    for (uint32_t k = 0; k < s->n_load; ++k)
                    {
                        if (s->load_ids[k] == w[i+3])
                        {
                            descriptor_var = s->load_vars[k];
                            break;
                        }
                    }
                    int vi = fs_var_index(s, descriptor_var);
                    STEREO_LOG(
                        "FS_IMAGE_DESCRIPTOR image=%u descriptorVar=%u set=%u binding=%u",
                        w[i+3],
                        descriptor_var,
                        vi >= 0 ? s->var_set[vi] : 999,
                        vi >= 0 ? s->var_binding[vi] : 999);
                }
                /* OpLoad of sampled/image/sampled-image objects */
                if (op == SpvOpLoad &&
                    wc >= 4 &&
                    s->n_load < FS_MAX_LOADS &&
                    fs_is_image_related_type(s, w[i+1]))
                {
                    uint32_t idx = s->n_load++;
                    s->load_ids[idx] = w[i+2];
                    s->load_vars[idx] =
                        fs_resolve_parameter_owner(
                            s,
                            w[i+3]);
                    STEREO_LOG(
                        "FS_LOAD_OWNER_RESOLVED load=%u source=%u owner=%u",
                        w[i+2],
                        w[i+3],
                        s->load_vars[idx]);
                    STEREO_LOG(
                        "FS_LOAD_RESOLVE_RESULT load=%u source=%u owner=%u",
                        w[i+2],
                        w[i+3],
                        s->load_vars[idx]);
                    /* Resolve function parameter ownership */
                    for (uint32_t p = 0; p < s->n_call; ++p)
                    {
                        if (s->call_params[p] == s->load_vars[idx])
                        {
                            s->load_vars[idx] =
                                s->call_args[p];
                            STEREO_LOG(
                                "FS_PARAM_RESOLVE param=%u descriptor=%u",
                                w[i+3],
                                s->call_args[p]);
                            break;
                        }
                    }
                    STEREO_LOG(
                        "FS_LOAD_MAP load=%u owner=%u tableIndex=%u",
                        s->load_ids[idx],
                        s->load_vars[idx],
                        idx);
                    int vi = fs_var_index(s, w[i+3]);
                    STEREO_LOG(
                        "FS_LOAD_SOURCE result=%u sourceVar=%u knownVar=%d set=%u binding=%u type=%u",
                        w[i+2],
                        w[i+3],
                        vi,
                        vi >= 0 ? s->var_set[vi] : 999,
                        vi >= 0 ? s->var_binding[vi] : 999,
                        vi >= 0 ? s->var_types[vi] : 999);
                    STEREO_LOG(
                        "FS_LOAD_TABLE image idx=%u id=%u type=%u var=%u",
                        idx,
                        s->load_ids[idx],
                        w[i+1],
                        s->load_vars[idx]);
                    STEREO_LOG(
                        "FS OpLoad IMAGE_TYPE: type=%u result=%u var=%u",
                        w[i+1],
                        w[i+2],
                        w[i+3]);
                }
                /* OpImage */
                if (op == SpvOpImage && wc >= 4)
                {
                    STEREO_LOG(
                        "FS OpImage result=%u image=%u",
                        w[i+2],
                        w[i+3]);
                     uint32_t descriptor_var = 0;
                     int found = 0;
                     for (uint32_t j = 0; j < s->n_load; ++j)
                     {
                         if (s->load_ids[j] == w[i+3])
                         {
                             descriptor_var = s->load_vars[j];
                             found = 1;
                             break;
                         }
                     }
                    STEREO_LOG(
                        "FS_OPIMAGE_DESCRIPTOR result=%u src=%u found=%d descriptorVar=%u imageRelated=%u",
                        w[i+2],
                        w[i+3],
                        found,
                        descriptor_var,
                        fs_is_image_related_type(s, w[i+1]));
                }
                /* OpCopyObject */
                if (op == SpvOpCopyObject && wc >= 4)
                {
                    STEREO_LOG(
                        "FS_COPY_OBJECT result=%u src=%u",
                        w[i+2],
                        w[i+3]);
                }
                /* OpSampledImage */
                if (op == SpvOpSampledImage && wc >= 5 &&
                    fs_id_in(s->si_ids, s->n_si, w[i+1]) &&
                    s->n_load < FS_MAX_LOADS)
                {
                    uint32_t idx = s->n_load++;
                    s->load_ids[idx] = w[i+2];
                    /* Preserve the originating descriptor variable so later
                     * OpImageSample* logging can recover the binding.
                     */
                    s->load_vars[idx] = 0;
                    for (uint32_t j = 0; j < idx; ++j)
                    {
                        if (s->load_ids[j] == w[i+3])
                        {
                            s->load_vars[idx] = s->load_vars[j];
                            STEREO_LOG(
                                "FS_LOAD_PROPAGATE sampledImage=%u descriptorVar=%u",
                                w[i+3],
                                s->load_vars[j]);
                            break;
                        }
                    }
                    if (s->load_vars[idx] == 0)
                    {
                        STEREO_LOG(
                            "FS_LOAD_PROPAGATE FAILED sampledImage=%u",
                            w[i+3]);
                    }
                    STEREO_LOG(
                        "FS OpSampledImage type=%u result=%u image=%u sampler=%u imagePatched=%u samplerPatched=%u",
                        w[i+1],
                        w[i+2],
                        w[i+3],
                        w[i+4],
                        fs_id_in(s->load_ids, s->n_load, w[i+3]),
                        fs_id_in(s->load_ids, s->n_load, w[i+4]));
                }
                /*
                 * Propagate descriptor ownership through image-producing
                 * instructions.
                 *
                 * Deferred renderers commonly do:
                 *
                 *   OpLoad          %196
                 *   OpImage         %198 %196
                 *   OpImageFetch    ...  %198
                 *
                 * so %198 must inherit %196's descriptor variable.
                 */
                if ((op == SpvOpImage ||
                    (op == SpvOpCopyObject &&
                    fs_is_image_related_type(s, w[i+1]))) &&
                    s->n_load < FS_MAX_LOADS)
                {
                    uint32_t src = w[i+3];
                    STEREO_LOG(
                        "FS_PROPAGATE_TRY op=%s(%u) src=%u dst=%u",
                        spv_op_name(op),
                        op,
                        src,
                        w[i+2]);
                    int propagated = 0;
                    for (uint32_t k = 0; k < s->n_load; ++k)
                    {
                        if (s->load_ids[k] == src)
                        {
                            STEREO_LOG(
                                "FS_PROPAGATE_SOURCE src=%u owner=%u",
                                src,
                                s->load_vars[k]);
                            int owner_vi = fs_var_index(
                                s,
                                s->load_vars[k]);
                            STEREO_LOG(
                                "FS_IMAGE_PROPAGATE_OWNER_CHECK src=%u dst=%u owner=%u known=%u",
                                src,
                                w[i+2],
                                s->load_vars[k],
                                owner_vi >= 0);
                            uint32_t idx = s->n_load++;
                            s->load_ids[idx]  = w[i+2];
                            s->load_vars[idx] = s->load_vars[k];
                            STEREO_LOG(
                                "FS_IMAGE_PROPAGATE op=%s(%u) src=%u dst=%u owner=%u srcIndex=%u",
                                spv_op_name(op),
                                op,
                                src,
                                w[i+2],
                                s->load_vars[k],
                                k);
                            propagated = 1;
                            break;
                        }
                    }
                    if (!propagated)
                    {
                        STEREO_LOG(
                            "FS_IMAGE_PROPAGATE_FAILED op=%s(%u) src=%u dst=%u",
                            spv_op_name(op),
                            op,
                            src,
                            w[i+2]);
                    }
                }
            }
            break;
        }
        i += wc;
    }
    /*
     * Resolve deferred function-call arguments after all
     * OpFunctionParameter instructions are known.
     */
    for (uint32_t c = 0; c < s->n_call; ++c)
    {
        STEREO_LOG(
            "FS_FIXUP_SCAN index=%u function=%u arg=%u",
            c,
            s->call_functions[c],
            s->call_args[c]);
        uint32_t fn = s->call_functions[c];
        uint32_t arg_index = s->call_params[c];
        for (uint32_t f = 0; f < s->n_function; ++f)
        {
            if (s->function_ids[f] != fn)
                continue;
            uint32_t p =
                s->function_param_start[f] + arg_index;
            if (p < s->n_param)
            {
                STEREO_LOG(
                    "FS_FUNCTION_ARG_FIXUP function=%u param=%u arg=%u",
                    fn,
                    s->param_ids[p],
                    s->call_args[c]);
                /*
                 * Now that call parameter ownership is known, rewrite any
                 * recorded load owners that still reference parameter IDs.
                 */
                for (uint32_t l = 0; l < s->n_load; ++l)
                {
                    for (uint32_t c = 0; c < s->n_call; ++c)
                    {
                        if (s->load_vars[l] == s->call_params[c])
                        {
                            STEREO_LOG(
                                "FS_LOAD_FINAL_RESOLVE load=%u param=%u owner=%u",
                                s->load_ids[l],
                                s->load_vars[l],
                                s->call_args[c]);
                
                            s->load_vars[l] = s->call_args[c];
                            break;
                        }
                    }
                }
                s->call_params[c] =
                    s->param_ids[p];
            }
        }
    }
    for (uint32_t i = 0; i < s->n_var; ++i)
    {
        STEREO_LOG(
            "FS_VAR_FINAL id=%u type=%u set=%u binding=%u",
            s->var_ids[i],
            s->var_types[i],
            s->var_set[i],
            s->var_binding[i]);
    }
    for (uint32_t i = 0; i < s->n_dec; ++i)
    {
        STEREO_LOG(
            "FS_DEC target=%u set=%u binding=%u",
            s->dec_target[i],
            s->dec_set[i],
            s->dec_binding[i]);
    }
}

static uint32_t fs_count_patches(const FsScan *s, const uint32_t *w, size_t c)
{
    uint32_t count = 0;
    bool in_func = false;

    for (size_t i = 5; i < c; )
    {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (!wc || i + wc > c)
            break;
        if (op == 54)
            in_func = true;
        if (in_func &&
            wc >= 5 &&
            (op == 87 || op == 88 || op == 89 || op == 90) &&
            fs_id_in(s->load_ids, s->n_load, w[i+3]))
        {
            STEREO_LOG(
                "FS_PATCH_COUNTER sample image=%u result=%u coord=%u total=%u",
                w[i+3],
                w[i+2],
                w[i+4],
                count + 1);
            count++;
        }
        /* OpImageFetch */
        if (in_func && op == 95 && wc >= 5)
        {
            uint32_t descriptor_var = 0;
            for (uint32_t k = 0; k < s->n_load; ++k)
            {
                if (s->load_ids[k] == w[i+3])
                {
                    descriptor_var = s->load_vars[k];
                    break;
                }
            }
            if (descriptor_var == 0)
            {
                STEREO_LOG(
                    "FS_FETCH_NO_DESCRIPTOR image=%u",
                    w[i+3]);
            }
            if (fs_binding_is_stereo_attachment(
                    s,
                    descriptor_var))
            {
                uint32_t binding = 0xffffffffu;
                for (uint32_t k = 0; k < s->n_var; ++k)
                {
                    if (s->var_ids[k] == descriptor_var)
                    {
                        binding = s->var_binding[k];
                        break;
                    }
                }
                STEREO_LOG(
                    "FS_SAMPLE_PATCH_APPLY descriptor=%u binding=%u",
                    descriptor_var,
                    binding);
                STEREO_LOG(
                    "FS_PATCH_COUNTER fetch image=%u result=%u coord=%u total=%u",
                    w[i+3],
                    w[i+2],
                    w[i+4],
                    count + 1);
                count++;
            }
        }
        i += wc;
    }
    return count;
}

bool spirv_patch_stereo_fs(
    const uint32_t *in, size_t in_c,
    uint32_t **out, size_t *out_c)
{
    if (!in || in_c < 5 || in[0] != SPIRV_MAGIC) return false;
    FsScan s;
    fs_prescan(&s, in, in_c);
    for (uint32_t i = 0; i < s.n_var; ++i)
    {
        if (s.var_binding[i] != 0xffffffffu)
        {
            STEREO_LOG(
                "FS_DESCRIPTOR_SUMMARY var=%u set=%u binding=%u type=%u",
                s.var_ids[i],
                s.var_set[i],
                s.var_binding[i],
                s.var_types[i]);
        }
    }
    if (s.n_img == 0 || !s.float_id) return false;

    uint32_t n_patches = fs_count_patches(&s, in, in_c);

    /* Allocate new IDs above current bound */
    uint32_t nid           = in[3];
    uint32_t new_int_id    = s.int_id        ? s.int_id        : nid++;
    uint32_t new_v3f_id    = s.v3float_id    ? s.v3float_id    : nid++;
    uint32_t new_v3i_id    = nid++;
    uint32_t new_pin_id    = s.ptr_int_in_id ? s.ptr_int_in_id : nid++;
    uint32_t new_vi_id     = s.vi_var_id     ? s.vi_var_id     : nid++;
    bool     is_new_vi     = (s.vi_var_id == 0);
    uint32_t samp_nid      = nid;
    uint32_t new_bound     = samp_nid + n_patches * 5 + 8;

    SpvBuf ob;
    if (!sb_init(&ob, in_c + 60 + (size_t)n_patches * 28)) return false;

    bool mv_added   = s.has_mv_cap;
    bool types_done = false;
    bool ep_done    = false;
    bool in_func    = false;

    /* Header */
    sb_push_n(&ob, in, 5);
    ob.w[3] = new_bound;

    for (size_t i = 5; i < in_c; ) {
        uint32_t op = in[i] & 0xffff, wc = in[i] >> 16;
        if (!wc || i + wc > in_c) break;

        if (in_func &&
            op >= SpvOpImageSampleImplicitLod &&
            op <= SpvOpImageSparseTexelsResident)
        {
            STEREO_LOG(
                "FS_IMAGE_OPCODE op=%u (%s) wc=%u resultType=%u result=%u",
                op,
                spv_op_name(op),
                wc,
                (wc >= 2) ? in[i + 1] : 0,
                (wc >= 3) ? in[i + 2] : 0);
        }

        /* Add MultiView capability before first non-capability instruction */
        if (!mv_added && op != 17) {
            uint32_t mv[] = { (2u<<16)|17, 4439 };
            sb_push_n(&ob, mv, 2);
            mv_added = true;
        }

        /* Modify OpEntryPoint: append new_vi_id to interface if we're adding it */
        if (op == 15 && !ep_done) {
            ep_done = true;
            if (is_new_vi) {
                sb_push(&ob, ((wc+1)<<16)|15);
                sb_push_n(&ob, &in[i+1], wc-1);
                sb_push(&ob, new_vi_id);
            } else {
                sb_push_n(&ob, &in[i], wc);
            }
            i += wc; continue;
        }

        /* Patch OpTypeImage: Dim=2D Arrayed=0 → Arrayed=1 (in-place word change) */
        if (op == 25 && wc >= 9 &&
            fs_id_in(s.img_ids, s.n_img, in[i+1]) &&
            in[i+5] == 0) {
            STEREO_LOG(
                "FS_IMAGE_PATCH_DETAIL type=%u sampled=%u dim=%u depth=%u arrayed=%u ms=%u format=%u",
                in[i+1],
                in[i+7],
                in[i+3],
                in[i+4],
                in[i+5],
                in[i+6],
                in[i+8]);
            STEREO_LOG(
                "FS_PATCH_IMAGE word=%zu type=%u dim=%u depth=%u arrayed=%u",
                i,
                in[i+1],
                in[i+3],
                in[i+4],
                in[i+5]);
            STEREO_LOG(
                "FS discovered image type id=%u depth=%u arrayed=%u sampled=%u",
                in[i+1],
                in[i+4],
                in[i+5],
                in[i+7]);
            STEREO_LOG(
                "FS_ARRAY_UPGRADE type=%u",
                in[i+1]);
            STEREO_LOG(
                "FS converting image type id=%u depth=%u arrayed=%u",
                in[i+1],
                in[i+4],
                in[i+5]);
            sb_push_n(&ob, &in[i], wc);
            ob.w[ob.n - wc + 5] = 1; /* Arrayed */
            i += wc;
            continue;
        }

        /* Inject new types + gl_ViewIndex variable before first OpFunction */
        if (op == 54 && !types_done) {
            types_done = true;
            in_func    = true;
            if (is_new_vi) {
                /* OpDecorate %vi BuiltIn ViewIndex */
                { uint32_t w[]={(4u<<16)|71, new_vi_id, 11, 4440};
                  sb_push_n(&ob,w,4); }
            }
            if (!s.int_id) {
                uint32_t w[]={(4u<<16)|21, new_int_id, 32, 1};
                sb_push_n(&ob,w,4); }
            if (!s.v3float_id) {
                uint32_t w[]={(4u<<16)|23, new_v3f_id, s.float_id, 3};
                sb_push_n(&ob,w,4); }
            {
                uint32_t w[]={(4u<<16)|23, new_v3i_id, new_int_id, 3};
                sb_push_n(&ob,w,4);
            }
            if (!s.ptr_int_in_id) {
                uint32_t w[]={(4u<<16)|32, new_pin_id, 1, new_int_id};
                sb_push_n(&ob,w,4); }
            if (is_new_vi) {
                /* OpVariable %ptr_int_in Input → %vi */
                uint32_t w[]={(4u<<16)|59, new_pin_id, new_vi_id, 1};
                sb_push_n(&ob,w,4); }
            sb_push_n(&ob, &in[i], wc);
            i += wc; continue;
        }
        if (op == 54) in_func = true;
        /*
         * Log fragment shader output stores.
         * OpStore operands:
         *   word[1] = target variable
         *   word[2] = stored value
         *
         * Used to identify SSAO/deferred lighting outputs.
         */
        if (in_func && op == 62 && wc >= 3)
        {
            uint32_t target = in[i+1];
            int vi = fs_var_index(&s, target);
            if (vi >= 0)
            {
                STEREO_LOG(
                    "FS_OUTPUT target=%u set=%u location=%u type=%u value=%u",
                    target,
                    s.var_set[vi],
                    s.var_binding[vi],
                    s.var_types[vi],
                    in[i+2]);
            }
            else
            {
                STEREO_LOG(
                    "FS_OUTPUT_UNKNOWN target=%u value=%u",
                    target,
                    in[i+2]);
            }
        }
        /* Extend 2D sampling coordinate to 3D for patched loads */
        if (in_func && wc >= 5 &&
            (op == 87 || op == 88 || op == 89 || op == 90) &&
            fs_id_in(s.load_ids, s.n_load, in[i+3]))
        {
            STEREO_LOG(
                "FS extending sample: op=%u sampledImage=%u coord=%u result=%u",
                op,
                in[i+3],
                in[i+4],
                in[i+2]);
            uint32_t coord_id = in[i+4];
            uint32_t descriptor_var = 0;
            for (uint32_t k = 0; k < s.n_load; ++k)
            {
                if (s.load_ids[k] == in[i+3])
                {
                    descriptor_var = s.load_vars[k];
                    int vi = fs_var_index(&s, descriptor_var);
                    STEREO_LOG(
                        "FS_SAMPLE_DESCRIPTOR image=%u descriptorVar=%u set=%u binding=%u",
                        in[i+3],
                        descriptor_var,
                        (vi >= 0) ? s.var_set[vi] : 0xffffffffu,
                        (vi >= 0) ? s.var_binding[vi] : 0xffffffffu);
                    STEREO_LOG(
                         "FS_SAMPLE_BINDING_DETAIL image=%u descriptor=%u binding=%u",
                         in[i+3],
                         descriptor_var,
                         (vi >= 0) ? s.var_binding[vi] : 0xffffffffu);
                    STEREO_LOG(
                        "FS_SAMPLE_MATCH image=%u load=%u var=%u",
                        in[i+3],
                        k,
                        descriptor_var);
                    break;
                }
            }
            if (!fs_binding_is_stereo_attachment(&s, descriptor_var))
            {
                STEREO_LOG(
                    "FS_SAMPLE_SKIP_MONO image=%u descriptor=%u",
                    in[i+3],
                    descriptor_var);
            
                sb_push_n(&ob, &in[i], wc);
                i += wc;
                continue;
            }
            STEREO_LOG(
                "FS_SAMPLE_PATCH_APPLY image=%u descriptor=%u",
                in[i+3],
                descriptor_var);

            uint32_t id_lv  = samp_nid++;
            uint32_t id_cvt = samp_nid++;
            uint32_t id_u   = samp_nid++;
            uint32_t id_v   = samp_nid++;
            uint32_t id_c3  = samp_nid++;

            /* OpLoad %int %vi → id_lv */
            { uint32_t w[]={(4u<<16)|61, new_int_id, id_lv, new_vi_id};
              sb_push_n(&ob,w,4); }
            /* OpConvertSToF %float id_lv → id_cvt */
            { uint32_t w[]={(4u<<16)|111, s.float_id, id_cvt, id_lv};
              sb_push_n(&ob,w,4); }
            /* OpCompositeExtract %float coord 0 → id_u */
            { uint32_t w[]={(5u<<16)|81, s.float_id, id_u, coord_id, 0};
              sb_push_n(&ob,w,5); }
            /* OpCompositeExtract %float coord 1 → id_v */
            { uint32_t w[]={(5u<<16)|81, s.float_id, id_v, coord_id, 1};
              sb_push_n(&ob,w,5); }
            /* OpCompositeConstruct %v3float id_u id_v id_cvt → id_c3 */
            { uint32_t w[]={(6u<<16)|80, new_v3f_id, id_c3, id_u, id_v, id_cvt};
              sb_push_n(&ob,w,6); }

            /* Emit modified sample instruction: word[4] = new coord */
            sb_push(&ob, in[i]);          /* opcode */
            sb_push(&ob, in[i+1]);        /* result type */
            sb_push(&ob, in[i+2]);        /* result id */
            sb_push(&ob, in[i+3]);        /* sampled image (unchanged) */
            sb_push(&ob, id_c3);          /* new 3D coordinate */
            if (wc > 5) sb_push_n(&ob, &in[i+5], wc-5); /* image operands */
            i += wc; continue;
        }
        if (in_func && op == 86 && wc >= 3)
        {
            STEREO_LOG(
                "FS_IMAGE imageResult=%u sampledImage=%u",
                in[i+2],
                in[i+3]);
        }
        /* Extend OpImageFetch ivec2 -> ivec3(x,y,ViewIndex) */
        if (in_func && op == 95 && wc >= 5)
        {
            STEREO_LOG(
                "FS_FETCH opcode image=%u coord=%u result=%u",
                in[i+3],
                in[i+4],
                in[i+2]);
            STEREO_LOG(
                "FS_FETCH_PATCH_ENTER image=%u result=%u",
                in[i+3],
                in[i+2]);
            uint32_t coord_id = in[i+4];
            uint32_t descriptor_var = 0;
            bool image_known = false;
            for (uint32_t k = 0; k < s.n_load; ++k)
            {
                if (s.load_ids[k] == in[i+3])
                {
                    descriptor_var = s.load_vars[k];
                    image_known = true;
                    STEREO_LOG(
                        "FS_FETCH_MATCH image=%u loadIndex=%u load=%u var=%u",
                        in[i+3],
                        k,
                        s.load_ids[k],
                        descriptor_var);
                    break;
                }
            }
            STEREO_LOG(
                "FS_FETCH_PATCH_DECISION image=%u known=%u descriptor=%u",
                in[i+3],
                image_known,
                descriptor_var);
            int found = 0;
            for (uint32_t k = 0; k < s.n_load; ++k)
            {
                if (s.load_ids[k] == in[i+3])
                {
                    STEREO_LOG(
                        "FS_FETCH_FOUND image=%u loadIndex=%u var=%u",
                        in[i+3],
                        k,
                        s.load_vars[k]);
                    found = 1;
                    break;
                }
            }
            if (!found)
            {
                STEREO_LOG(
                    "FS_FETCH_UNKNOWN image=%u",
                    in[i+3]);
            }
            int vi = fs_var_index(&s, descriptor_var);
            
            STEREO_LOG(
                "FS_FETCH_DESCRIPTOR image=%u descriptorVar=%u set=%u binding=%u",
                in[i+3],
                descriptor_var,
                (vi >= 0) ? s.var_set[vi] : 0xffffffffu,
                (vi >= 0) ? s.var_binding[vi] : 0xffffffffu);
            if (in[i+3] == 47)
            {
                STEREO_LOG(
                    "FS_TRACE_IMAGE47 result=%u coord=%u",
                    in[i+2],
                    coord_id);
            }
            if (!fs_binding_is_stereo_attachment(&s, descriptor_var))
            {
                STEREO_LOG(
                    "FS_FETCH_SKIP_MONO image=%u descriptor=%u binding_not_stereo",
                    in[i+3],
                    descriptor_var);
                sb_push_n(&ob, &in[i], wc);
                i += wc;
                continue;
            }
            STEREO_LOG(
                "FS_FETCH_STEREO_PATCH image=%u descriptorVar=%u coord=%u",
                in[i+3],
                descriptor_var,
                in[i+4]);
            uint32_t id_lv = samp_nid++;
            uint32_t id_x  = samp_nid++;
            uint32_t id_y  = samp_nid++;
            uint32_t id_c3 = samp_nid++;

            { uint32_t w[]={(4u<<16)|61, new_int_id, id_lv, new_vi_id};
              sb_push_n(&ob,w,4); }

            { uint32_t w[]={(5u<<16)|81, new_int_id, id_x, coord_id, 0};
              sb_push_n(&ob,w,5); }

            { uint32_t w[]={(5u<<16)|81, new_int_id, id_y, coord_id, 1};
              sb_push_n(&ob,w,5); }

            { uint32_t w[]={(6u<<16)|80, new_v3i_id, id_c3,
                            id_x, id_y, id_lv};
              sb_push_n(&ob,w,6); }

            sb_push(&ob, in[i]);
            sb_push(&ob, in[i+1]);
            sb_push(&ob, in[i+2]);
            sb_push(&ob, in[i+3]);
            sb_push(&ob, id_c3);

            if (wc > 5)
                sb_push_n(&ob, &in[i+5], wc - 5);
            STEREO_LOG(
                "FS_FETCH_PATCH_DONE image=%u newCoord=%u",
                in[i+3],
                id_c3);
            i += wc;
            continue;
        }

        sb_push_n(&ob, &in[i], wc);
        i += wc;
    }

    ob.w[3] = samp_nid + 1;
    *out   = ob.w;
    *out_c = ob.n;
    STEREO_LOG("FS patched: %u 2D img types→arr, %u samples extended, bound %u→%u",
               s.n_img, n_patches, in[3], ob.w[3]);
    return true;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
static bool is_patchable_spv(const uint32_t *w, size_t c)
{
    if (c<5||w[0]!=SPIRV_MAGIC) return false;
    for (size_t i=5;i<c;) {
        uint32_t op=w[i]&0xffff, wc=w[i]>>16; if (!wc||i+wc>c) break;
        if (op==SpvOpEntryPoint&&wc>=2) {
            uint32_t e=w[i+1];
            return e==SpvExecVertex||e==SpvExecGeometry||e==SpvExecTessEval||e==4/*Fragment*/;
        }
        i+=wc;
    }
    return false;
}

static StereoShaderCache *cache_find(StereoDevice *sd, VkShaderModule h)
{
    for (uint32_t i=0;i<sd->shader_cache_count;i++)
        if (sd->shader_cache[i].handle==h) return &sd->shader_cache[i];
    return NULL;
}

static void cache_add(StereoDevice *sd, VkShaderModule h,
                      const uint32_t *spv, size_t words) {
    if (sd->shader_cache_count>=MAX_SHADER_CACHE) return;
    uint32_t *cp=malloc(words*4); if (!cp) return;
    memcpy(cp,spv,words*4);
    StereoShaderCache *e=&sd->shader_cache[sd->shader_cache_count++];
    e->handle=h; e->spv=cp; e->words=words;
}
static void cache_remove(StereoDevice *sd, VkShaderModule h) {
    for (uint32_t i=0;i<sd->shader_cache_count;i++)
        if (sd->shader_cache[i].handle==h) {
            free(sd->shader_cache[i].spv);
            sd->shader_cache[i]=sd->shader_cache[--sd->shader_cache_count];
            return; }
}

/* ── vkCreateShaderModule ─────────────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCI,
                          const VkAllocationCallbacks *pAlloc, VkShaderModule *pSM)
{
    StereoDevice *sd=stereo_device_from_handle(device);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    VkResult res=sd->real.CreateShaderModule(sd->real_device,pCI,pAlloc,pSM);
    if (res!=VK_SUCCESS) return res;
    if (!sd->stereo.enabled) return VK_SUCCESS;
    const uint32_t *spv=(const uint32_t*)pCI->pCode;
    size_t wc=pCI->codeSize/4;
    if (is_patchable_spv(spv,wc)) cache_add(sd,*pSM,spv,wc);
    return VK_SUCCESS;
}

/* ── vkCreateGraphicsPipelines ───────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateGraphicsPipelines(VkDevice device, VkPipelineCache pc,
    uint32_t N, const VkGraphicsPipelineCreateInfo *pCI,
    const VkAllocationCallbacks *pAlloc, VkPipeline *pP)
{
    StereoDevice *sd=stereo_device_from_handle(device);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    STEREO_LOG("PIPE_IN_RAW N=%u pCI=%p first=%p renderPass=%p stageCount=%u pNext=%p",
               N,
               (void*)pCI,
               (N > 0 ? (void*)pCI[0].renderPass : NULL),
               (N > 0 ? (void*)pCI[0].renderPass : NULL),
               (N > 0 ? pCI[0].stageCount : 0),
               (N > 0 ? pCI[0].pNext : NULL));

    STEREO_LOG(
        "PIPE_CREATE_BEGIN N=%u multiview=%d enabled=%d",
        N,
        sd->stereo.multiview,
        sd->stereo.enabled);
    if (!sd->stereo.enabled)
        return sd->real.CreateGraphicsPipelines(sd->real_device,pc,N,pCI,pAlloc,pP);

    VkShaderModule                   *tmp_mod = calloc(N, sizeof(VkShaderModule));
    VkPipelineShaderStageCreateInfo **tst     = calloc(N, sizeof(void*));
    VkGraphicsPipelineCreateInfo     *infos   = malloc(N * sizeof(*infos));
    if (!tmp_mod||!tst||!infos) {
        free(tmp_mod); free(tst); free(infos);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    memcpy(infos, pCI, N * sizeof(*infos));
    for (uint32_t p = 0; p < N; p++) {
        STEREO_LOG(
            "PIPE_IN p=%u rp=%p stageCount=%u vs=%d tcs=%d tes=%d pNext=%p",
            p,
            (void*)pCI[p].renderPass,
            pCI[p].stageCount,
            (pCI[p].pVertexInputState != NULL),
            0,
            0,
            pCI[p].pNext);
    }
    const char *dump = stereo_getenv("VKS3D_DUMP_SPIRV");
    static int  dump_n = 0;
    float lo=sd->stereo.left_eye_offset, ro=sd->stereo.right_eye_offset,
          conv=sd->stereo.convergence;

    STEREO_LOG(
        "[PATCH] lo=%f ro=%f flip=%d",
        lo,
        ro,
        sd->stereo.flip_eyes);
    for (uint32_t p=0; p<N; p++) {
        const VkGraphicsPipelineCreateInfo *ci=&pCI[p];
        StereoPipelineInfo *info =
            add_pipeline_info(sd);
        const VkBaseInStructure *base =
            (const VkBaseInStructure*)ci->pNext;
        uint32_t view_mask = 0;
        /* ── Safety: Vulkan 1.3 dynamic rendering pipelines may not use pNext ── */
        while (base)
        {
            if (base->sType ==
                VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
            {
                const VkPipelineRenderingCreateInfo *ri =
                 (const VkPipelineRenderingCreateInfo*)base;
                VkPipelineRenderingCreateInfo *rw =
                 (VkPipelineRenderingCreateInfo*)base;
                
                /* Dynamic rendering path: if stereo is enabled and the app left
                 * viewMask at 0, promote it to 0x3 so the pipeline is actually
                 * created for multiview. */
                if (sd->stereo.multiview && rw->viewMask == 0) {
                 STEREO_LOG(
                  "PIPE_RENDERING_UPGRADE p=%u viewMask 0x0->0x3 colors=%u depth=%u stencil=%u",
                  p,
                  ri->colorAttachmentCount,
                  ri->depthAttachmentFormat,
                  ri->stencilAttachmentFormat);
                 rw->viewMask = 0x3;
                }
                view_mask = rw->viewMask;
                STEREO_LOG(
                    "PIPE_RENDERING_CAPTURE p=%u viewMask=0x%x colors=%u depth=%u stencil=%u",
                    p,
                    rw->viewMask,
                    ri->colorAttachmentCount,
                    ri->depthAttachmentFormat,
                    ri->stencilAttachmentFormat);
            }
            base = base->pNext;
        }
        if (!ci || ci->stageCount == 0 || !ci->pStages) {
            STEREO_LOG(
                "PIPE_EMPTY_STAGE_PIPELINE p=%u rp=%p pNext=%p stageCount=%u pStages=%p isUI=%d isComputeLike=%d",
                p,
                ci ? (void*)ci->renderPass : NULL,
                ci ? (void*)ci->pNext : NULL,
                ci ? ci->stageCount : 0,
                ci ? (void*)ci->pStages : NULL,
                (ci && ci->pVertexInputState == NULL),
                (ci && ci->stageCount == 0));
        }
        if (!ci ||
            ci->stageCount == 0 ||
            !ci->pStages)
        {
            STEREO_LOG("PIPE_INVALID p=%u ci=%p stageCount=%u pStages=%p renderPass=%p",
                       p,
                       (void*)ci,
                       ci ? ci->stageCount : 0,
                       ci ? (void*)ci->pStages : NULL,
                       ci ? (void*)ci->renderPass : NULL);
            continue;
        }
        bool has_vs=false, has_tcs=false, has_tes=false;
        uint32_t vs_stage=~0u, tes_stage=~0u;
        for (uint32_t s=0;s<ci->stageCount;s++) {
            VkShaderStageFlagBits st=ci->pStages[s].stage;
            if (st==VK_SHADER_STAGE_VERTEX_BIT)
                { has_vs=true; vs_stage=s; }
            if (st==VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
                has_tcs=true;
            if (st==VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
                { has_tes=true; tes_stage=s; }
        }

        /* ── Determine if this pipeline's render pass has multiview ──────
         * gl_ViewIndex is 0 in non-multiview passes.  Patching VS/TES there
         * bakes in left_eye_offset for ALL draws → deferred G-buffer / shadow
         * passes render from left-eye-only perspective → monoscopic output.
         * Leave non-multiview pass shaders unpatched so G-buffer, shadow maps,
         * and post-fx all render from the CENTER perspective; the multiview
         * final (swapchain) pass applies per-eye shift → image-space stereo
         * for deferred content with shadows/lights/bloom properly aligned.   */
        StereoRenderPassInfo *rpi = NULL;
        bool in_mv_rp = false;
        if (ci->renderPass != VK_NULL_HANDLE) {
            rpi = stereo_rp_lookup(sd, ci->renderPass);
            /* Render-pass pipelines are multiview only if the render pass itself
             * was created with multiview support. Shadow passes must stay mono. */
            in_mv_rp =
                (rpi && rpi->has_multiview) ||
                ((view_mask & 0x3) != 0);
        }
        else if (sd->stereo.multiview && (view_mask & 0x3) != 0) {
        /* VK 1.3 dynamic rendering: no renderPass handle, but we already
         * upgraded VkPipelineRenderingCreateInfo.viewMask above. Treat it
         * as multiview so VS/TES patching still runs. */
        in_mv_rp = true;
        }
        STEREO_LOG(
            "PIPE_DECISION p=%u rp=%p rpi=%p in_mv=%u view_mask=0x%x stages=%u has_vs=%u has_tes=%u quad=%u",
            p,
            (void*)ci->renderPass,
            (void*)rpi,
            (unsigned)in_mv_rp,
            view_mask,
            (unsigned)ci->stageCount,
            (unsigned)has_vs,
            (unsigned)has_tes,
            (!ci->pVertexInputState ||
             ci->pVertexInputState->vertexBindingDescriptionCount == 0));

        for (uint32_t fs_dbg_i = 0; fs_dbg_i < ci->stageCount; fs_dbg_i++) {
            if (ci->pStages[fs_dbg_i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                StereoShaderCache *fs_dbg =
                    cache_find(sd, ci->pStages[fs_dbg_i].module);
                if (fs_dbg) {
                    STEREO_LOG(
                        "ALL_FS_SHADER p=%u hash=%016llx words=%zu module=%p quad=%u in_mv=%u",
                        p,
                        (unsigned long long)hash_spv(fs_dbg->spv, fs_dbg->words),
                        fs_dbg->words,
                        (void*)ci->pStages[fs_dbg_i].module,
                        (!ci->pVertexInputState ||
                         ci->pVertexInputState->vertexBindingDescriptionCount == 0),
                        (unsigned)in_mv_rp);
                }
            }
        }

        /* ── PATCH 3: Pipeline multiview FIXED (NO pipeline struct exists) ─────────────── */
        /* Multiview is render-pass driven ONLY.
         * Pipeline pNext must NOT contain VkPipelineMultiviewCreateInfo (invalid Vulkan API). */
        if (in_mv_rp) {
            if (rpi && rpi->mv_handle) {
                STEREO_LOG(
                    "Pipe %u: MV RP detected (stageCount=%u) - using MV render pass %p",
                    p,
                    ci->stageCount,
                    (void*)rpi->mv_handle);
                /* render-pass pipeline path only */
                infos[p].renderPass = rpi->mv_handle;
            } else {
                STEREO_LOG(
                    "Pipe %u: dynamic rendering multiview detected (stageCount=%u) - no renderPass swap",
                    p,
                    ci->stageCount);
                /* VK 1.3 dynamic rendering: keep infos[p].renderPass as-is */
            }
        }

        if (!in_mv_rp)
        {
            STEREO_LOG(
                "Pipe %u: rp=%p not multiview (VS=%d TES=%d stages=%u)",
                p,
                (void*)(uintptr_t)ci->renderPass,
                has_vs,
                has_tes,
                ci->stageCount);

            /* IMPORTANT:
             * Do NOT patch renderpass-based multiview logic for clearly mono pipelines
             * BUT still allow FS quad / UI heuristics to run later
             */
            goto PIPE_DECISION_CONTINUE;
        }

        /* Substitute multiview render pass for pipeline compilation.
         * Pipelines must be compiled against the MV render pass so the driver
         * enables multiview optimisation and gl_ViewIndex receives the real
         * per-view index (0 or 1).  Render-pass compatibility rules allow these
         * pipelines to be used with both MV and non-MV framebuffers since
         * viewMask is not part of the compatibility criteria. */

        if (rpi && rpi->mv_handle && rpi->has_multiview && in_mv_rp)
        {
            /* Render-pass path only; dynamic rendering has no renderPass to swap. */
            infos[p].renderPass = rpi->mv_handle;
        }

        /* ── Full-screen quad detection ──────────────────────────────────
         * Pipelines with no vertex input bindings are full-screen quads used
         * by deferred lighting, SSAO, bloom, TAA, etc.  Their FS samples from
         * G-buffer / render-target textures (all upgraded to 2D_ARRAY by
         * stereo_CreateImage).  We patch the FS to use sampler2DArray +
         * gl_ViewIndex so each eye reads its own G-buffer layer.
         * The VS of a quad must NOT be patched — shifting the quad position
         * would prevent it covering the full screen for one eye.
         * Geometry pipelines (has vertex input) use Path A/B VS patching. */
        bool is_quad = !ci->pVertexInputState ||
                       ci->pVertexInputState->vertexBindingDescriptionCount == 0;

        if (is_quad && ci->stageCount > 0) {
            /* Find FS stage */
            uint32_t fs_s = ~0u;
            for (uint32_t s2 = 0; s2 < ci->stageCount; s2++)
                if (ci->pStages[s2].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
                    { fs_s = s2; break; }
            if (fs_s == ~0u) {
                STEREO_LOG("Pipe %u: quad but no FS stage", p);
                continue;
            }
            /*
             * Log fullscreen quad FS identity.
             * Used to identify SSAO/deferred/post-process shaders.
             */
            StereoShaderCache *fs_dbg =
                cache_find(sd, ci->pStages[fs_s].module);
            if (fs_dbg) {
                STEREO_LOG(
                    "QUAD_FS_SHADER p=%u hash=%016llx words=%zu module=%p",
                    p,
                    (unsigned long long)hash_spv(fs_dbg->spv, fs_dbg->words),
                    fs_dbg->words,
                    (void*)ci->pStages[fs_s].module);
            } else {
                STEREO_LOG(
                    "QUAD_FS_SHADER p=%u module=%p NOT_CACHED",
                    p,
                    (void*)ci->pStages[fs_s].module);
            }
            StereoShaderCache *e = cache_find(sd, ci->pStages[fs_s].module);
            if (!e) {
                STEREO_LOG("Pipe %u: quad FS not cached (stageCount=%u)", p, ci->stageCount);
                continue;
            }
            uint64_t spv_hash = hash_spv(e->spv, e->words);
            STEREO_LOG(
                "SHADER_MODULE stage=FS hash=%016llx words=%zu module=%p",
                (unsigned long long)spv_hash,
                e->words,
                (void*)ci->pStages[fs_s].module);
            STEREO_LOG(
                "PATCH hash=%016llx words=%zu module=%p vs_stage=%u",
                (unsigned long long)spv_hash,
                e->words,
                (void*)(has_vs ? ci->pStages[vs_stage].module : VK_NULL_HANDLE),
                vs_stage);
            if (dump) {
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx-fs.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f = fopen(dp, "wb");
                if (f) {
                    fwrite(e->spv,4,e->words,f);
                    fclose(f);
                }
            }
            uint32_t *patched = NULL; size_t pc2 = 0;
            STEREO_LOG(
                "FS_PATCH_BEGIN hash=%016llx",
                (unsigned long long)spv_hash);
            STEREO_LOG(
                "PATCHING_FS hash=%016llx",
                (unsigned long long)spv_hash);
            if (!spirv_patch_stereo_fs(e->spv, e->words, &patched, &pc2)) {
                STEREO_LOG("Pipe %u: FS patch skipped (no 2D samplers — material-only?)", p);
                continue;
            }
            if (dump) {
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx+fs.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f=fopen(dp,"wb");
                if (f) {
                    fwrite(patched,4,pc2,f);
                    fclose(f);
                }
            }
            VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                NULL,0,pc2*4,patched};
            VkShaderModule tmp=VK_NULL_HANDLE;
            VkResult mr=sd->real.CreateShaderModule(sd->real_device,&smci,NULL,&tmp);
            spirv_patched_free(patched);
            if (mr!=VK_SUCCESS) {
                STEREO_ERR("Pipe %u: quad FS module err %d",p,mr); continue; }
            uint32_t sc2=ci->stageCount;
            VkPipelineShaderStageCreateInfo *st=malloc(sc2*sizeof(*st));
            if (!st) { sd->real.DestroyShaderModule(sd->real_device,tmp,NULL); continue; }
            memcpy(st,ci->pStages,sc2*sizeof(*st));
            st[fs_s].module = tmp;
            infos[p].pStages = st;
            tmp_mod[p] = tmp;
            tst[p] = st;
            STEREO_LOG(
                "PATCHED_STAGE PathFS p=%u stage=%u orig=%p patched=%p",
                p,
                fs_s,
                (void *)ci->pStages[fs_s].module,
                (void *)tmp);
            STEREO_LOG(
                "Pipe %u: Path FS — quad sampler2DArray patch (%u stages)",
                p,
                sc2);
            continue;
        }

        /* ── Path A: patch existing TES ──────────────────────────────── */
        if (has_tes && tes_stage!=~0u) {
            StereoShaderCache *e=cache_find(sd, ci->pStages[tes_stage].module);
            if (!e) { STEREO_LOG("Pipe %u PathA: TES not cached",p); continue; }
            STEREO_LOG(
                "SHADER_MODULE stage=TES hash=%016llx words=%zu module=%p",
                (unsigned long long)hash_spv(e->spv, e->words),
                e->words,
                (void*)ci->pStages[tes_stage].module);
            if (dump) {
                uint64_t spv_hash = hash_spv(e->spv, e->words);
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx-ts.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f = fopen(dp, "wb");
                if (f) {
                    fwrite(e->spv,4,e->words,f);
                    fclose(f);
                }
            }
            uint32_t *patched=NULL; size_t pc2=0;
            STEREO_LOG(
                "[CALL A] lo=%f ro=%f conv=%f flip=%d",
                lo,
                ro,
                conv,
                sd->stereo.flip_eyes);
                StereoDebugCtx dbgA = {
                    p,
                    ci->renderPass,
                    in_mv_rp,
                    (uint32_t)VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                    0,
                    0
                };

                if (!spirv_patch_stereo_vertex(
                        &sd->stereo,
                        e->spv, e->words,
                        &patched, &pc2,
                        lo, ro, conv,
                        true,
                        &dbgA))
                {
                STEREO_LOG("TES patch failed");

                if (dump && patched && pc2) {
                    uint64_t spv_hash = hash_spv(e->spv, e->words);
                    char dp[512];
                    _snprintf(
                        dp,
                        sizeof(dp)-1,
                        "%s\\%016llx-ts_failed.spv",
                        dump,
                        (unsigned long long)spv_hash);
                    FILE *f=fopen(dp,"wb");
                    if (f) {
                        fwrite(patched,4,pc2,f);
                        fclose(f);
                    }
                }
                STEREO_LOG("Pipe %u PathA: patch failed",p);
                continue;
            }
            if (dump) {
                uint64_t spv_hash = hash_spv(e->spv, e->words);
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx+ts.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f=fopen(dp,"wb");
                if (f) {
                    fwrite(patched,4,pc2,f);
                    fclose(f);
                }
            }
            VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                NULL,0,pc2*4,patched};
            VkShaderModule tmp=VK_NULL_HANDLE;
            VkResult mr=sd->real.CreateShaderModule(sd->real_device,&smci,NULL,&tmp);
            spirv_patched_free(patched);
            if (mr!=VK_SUCCESS) {
                STEREO_ERR("Pipe %u PathA: module err %d",p,mr); continue; }
            uint32_t sc=ci->stageCount;
            VkPipelineShaderStageCreateInfo *st=malloc(sc*sizeof(*st));
            if (!st) { sd->real.DestroyShaderModule(sd->real_device,tmp,NULL); continue; }
            memcpy(st,ci->pStages,sc*sizeof(*st));
            st[tes_stage].module = tmp;
            infos[p].pStages = st;
            tmp_mod[p] = tmp;
            tst[p] = st;
            STEREO_LOG(
                "PATCHED_STAGE PathA p=%u stage=%u orig=%p patched=%p",
                p,
                tes_stage,
                (void *)ci->pStages[tes_stage].module,
                (void *)tmp);
            STEREO_LOG(
                "Pipe %u: Path A — TES patched (gl_ViewIndex)",
                p);
            continue;
        }

        STEREO_LOG(
        "PATHB_GATE p=%u in_mv=%d has_vs=%d has_tcs=%d vs_stage=%u",
        p,
        in_mv_rp,
        has_vs,
        has_tcs,
        vs_stage);
        /* ── Path B: patch VS with gl_ViewIndex ──────────────────────────
         * Only patch actual multiview render passes.
         * Non-multiview passes include deferred G-buffer, shadow, SSAO,
         * and post-processing passes that must remain center-eye.
         */
        if (in_mv_rp &&
            ci->stageCount > 0 &&
            has_vs &&
            !has_tcs &&
            vs_stage!=~0u) {
            StereoShaderCache *e=cache_find(sd, ci->pStages[vs_stage].module);
            if (!e) { STEREO_LOG("Pipe %u PathB: VS not cached",p); continue; }
            STEREO_LOG(
                "SHADER_MODULE stage=VS hash=%016llx words=%zu module=%p",
                (unsigned long long)hash_spv(e->spv, e->words),
                e->words,
                (void*)ci->pStages[vs_stage].module);
            STEREO_LOG(
                "VS_CONTEXT hash=%016llx rp=%p mv=%d color=%p depth=%d",
                (unsigned long long)hash_spv(e->spv, e->words),
                (void*)ci->renderPass,
                in_mv_rp,
                (void*)ci->renderPass,
                (ci->pDepthStencilState != NULL));
            if (dump) {
                uint64_t spv_hash = hash_spv(e->spv, e->words);
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx-vs.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f = fopen(dp, "wb");
                if (f) {
                    fwrite(e->spv, 4, e->words, f);
                    fclose(f);
                }
            }
            uint32_t *patched=NULL; size_t pc2=0;
            STEREO_LOG(
                "[CALL B] lo=%f ro=%f conv=%f flip=%d",
                lo,
                ro,
                conv,
                sd->stereo.flip_eyes);
            STEREO_LOG(
                "PATCH_CONSTS lo=%f ro=%f conv=%f",
                lo,
                ro,
                conv);
            STEREO_LOG(
                "[CALL B] multiview=%d pass_exists=%d",
                sd->stereo.multiview,
                sd->multiview_pass_exists);
            STEREO_LOG(
                "PathB candidate module=%p words=%zu",
                (void*)ci->pStages[vs_stage].module,
                e->words);
            StereoDebugCtx dbgB = {
                p,
                ci->renderPass,
                in_mv_rp,
                (uint32_t)VK_SHADER_STAGE_VERTEX_BIT,
                0,
                9
            };

            if (!spirv_patch_stereo_vertex(
                    &sd->stereo,
                    e->spv, e->words,
                    &patched, &pc2,
                    lo, ro, conv,
                    /*inj_vi=*/true,
                    &dbgB)) {
                STEREO_LOG("Pipe %u PathB: VS patch failed",p); continue; }
            if (dump) {
                uint64_t spv_hash = hash_spv(e->spv, e->words);
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx+vs.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f=fopen(dp,"wb");
                if (f) {
                    fwrite(patched,4,pc2,f);
                    fclose(f);
                }
            }
            VkShaderModuleCreateInfo smci={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                NULL,0,pc2*4,patched};
            VkShaderModule tmp=VK_NULL_HANDLE;
            VkResult mr=sd->real.CreateShaderModule(sd->real_device,&smci,NULL,&tmp);
            spirv_patched_free(patched);
            if (mr!=VK_SUCCESS) {
                STEREO_ERR("Pipe %u PathB: VS module err %d",p,mr); continue; }
            uint32_t sc=ci->stageCount;
            VkPipelineShaderStageCreateInfo *st=malloc(sc*sizeof(*st));
            if (!st) { sd->real.DestroyShaderModule(sd->real_device,tmp,NULL); continue; }
            memcpy(st,ci->pStages,sc*sizeof(*st));
            st[vs_stage].module = tmp;
            infos[p].pStages = st;
            tmp_mod[p] = tmp;
            tst[p] = st;
            STEREO_LOG(
                "PATCHED_STAGE PathB p=%u stage=%u orig=%p patched=%p",
                p,
                vs_stage,
                (void *)ci->pStages[vs_stage].module,
                (void *)tmp);
            STEREO_LOG(
                "Pipe %u: Path B — VS gl_ViewIndex patch",
                p);
            continue;
        }

        STEREO_LOG("Pipe %u: no patchable VS/TES stage (stageCount=%u has_vs=%d has_tes=%d has_tcs=%d) — not patched",
                   p, ci->stageCount, has_vs, has_tes, has_tcs);
    }

    PIPE_DECISION_CONTINUE:
    /* ── PATCH 5: RenderPass-based multiview binding ─────────────── */
    for (uint32_t p = 0; p < N; p++) {
        StereoRenderPassInfo *rpi = NULL;
        if (pCI[p].renderPass != VK_NULL_HANDLE)
            rpi = stereo_rp_lookup(sd, pCI[p].renderPass);
        STEREO_LOG(
            "PIPE_RP p=%u ci_rp=%p rpi=%p has_mv=%u mv=%p",
            p,
            (void*)pCI[p].renderPass,
            (void*)rpi,
            rpi ? (unsigned)rpi->has_multiview : 0,
            rpi ? (void*)rpi->mv_handle : NULL);
        if (rpi && rpi->has_multiview) {
            STEREO_LOG("Pipe %u: binding MV render pass %p", p, (void*)rpi->mv_handle);
            infos[p].renderPass = rpi->mv_handle;
        }
    }
    for (uint32_t p = 0; p < N; p++) {
        STEREO_LOG(
            "PIPE_FINAL p=%u ci_rp=%p final_rp=%p stages=%u",
            p,
            (void*)pCI[p].renderPass,
            (void*)infos[p].renderPass,
            infos[p].stageCount);
    }
    for (uint32_t p = 0; p < N; ++p)
    {
        STEREO_LOG(
            "PIPE_CREATE pipeline=%u renderPass=%p subpass=%u",
            p,
            infos[p].renderPass,
            infos[p].subpass);
        for (uint32_t s = 0; s < infos[p].stageCount; s++)
        {
            STEREO_LOG(
                "PIPE_STAGE p=%u stage=%u vkstage=0x%x module=%p patched_tmp=%u",
                p,
                s,
                infos[p].pStages[s].stage,
                (void *)infos[p].pStages[s].module,
                (unsigned)(
                    tmp_mod[p] != VK_NULL_HANDLE &&
                    infos[p].pStages[s].module == tmp_mod[p]));
        }
    }
    VkResult res=sd->real.CreateGraphicsPipelines(sd->real_device,pc,N,infos,pAlloc,pP);
    for (uint32_t p = 0; p < N; p++) {
        STEREO_LOG(
            "PIPE_CREATED pipe=%p result=%d rp=%p orig_rp=%p stages=%u",
            (res == VK_SUCCESS) ? (void*)pP[p] : NULL,
            res,
            (void*)infos[p].renderPass,
            (void*)pCI[p].renderPass,
            infos[p].stageCount);
        if (res == VK_SUCCESS)
        {
            StereoPipelineInfo *info =
                add_pipeline_info(sd);

            if (info)
            {
                info->pipeline = pP[p];

                info->original_renderpass =
                    pCI[p].renderPass;

                info->mv_renderpass =
                    infos[p].renderPass;

                info->stage_count =
                    infos[p].stageCount;

                info->is_quad =
                    (!pCI[p].pVertexInputState ||
                     pCI[p].pVertexInputState->vertexBindingDescriptionCount == 0);
                
                info->vertex_binding_count =
                    pCI[p].pVertexInputState ?
                    pCI[p].pVertexInputState->vertexBindingDescriptionCount : 0;

                info->view_mask = 0; /* default */

                for (uint32_t s = 0; s < infos[p].stageCount; s++)
                {
                    const VkPipelineShaderStageCreateInfo *st =
                        &infos[p].pStages[s];

                    if (st->stage == VK_SHADER_STAGE_VERTEX_BIT)
                    {
                        info->vs_module = st->module;
                        info->patched_vs =
                            (tmp_mod[p] != VK_NULL_HANDLE &&
                             st->module == tmp_mod[p]);
                    }

                    if (st->stage == VK_SHADER_STAGE_FRAGMENT_BIT)
                    {
                        info->fs_module = st->module;
                        info->patched_fs =
                            (tmp_mod[p] != VK_NULL_HANDLE &&
                             st->module == tmp_mod[p]);
                    }
                }
            }

            STEREO_LOG(
                "PIPE_INFO pipe=%p rp=%p orig_rp=%p stages=%u",
                (void*)pP[p],
                (void*)infos[p].renderPass,
                (void*)pCI[p].renderPass,
                infos[p].stageCount);
        }
    }
    STEREO_LOG(
        "PIPE_CREATE_END result=%d multiview_pass_exists=%d",
        res,
        sd->multiview_pass_exists);
    for (uint32_t p=0;p<N;p++) {
        if (tmp_mod[p]) {
            if (sd->tmp_module_count<MAX_TMP_MODULES)
                sd->tmp_modules[sd->tmp_module_count++]=tmp_mod[p];
            else
                sd->real.DestroyShaderModule(sd->real_device,tmp_mod[p],NULL);
        }
        free(tst[p]);
    }
    free(tmp_mod); free(tst); free(infos);
    return res;
}

/* ── vkDestroyShaderModule ───────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_DestroyShaderModule(VkDevice device, VkShaderModule sm,
                           const VkAllocationCallbacks *pAlloc)
{
    StereoDevice *sd=stereo_device_from_handle(device);
    if (!sd) return;
    cache_remove(sd,sm);
    sd->real.DestroyShaderModule(sd->real_device,sm,pAlloc);
}
