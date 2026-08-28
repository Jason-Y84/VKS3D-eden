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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stereo_icd.h"
#include "tes_inject.h"

/* Pull SPIR-V opcodes / constants from the Vulkan SDK's bundled SPIR-V
 * headers (spirv-headers/spirv.h).  This avoids depending on the
 * third_party/SPIRV-Headers git submodule while still giving us every
 * Spv* token used below.  The Vulkan SDK's Include directory is already
 * on the include path via ${Vulkan_INCLUDE_DIRS}. */
#include <spirv-headers/spirv.h>

#define SpvExecVertex           0
#define SpvExecTessEval         2
#define SpvExecGeometry         3
#define SpvStorageOutput        3
#define SpvStorageInput         1
#define SPIRV_MAGIC             0x07230203u

/* SPIR-V storage class fallbacks.  These are always stable and
 * independent of which spirv.h header (if any) the build picks up.
 * We deliberately do NOT provide opcode fallbacks here to avoid
 * colliding with whatever numeric values the official spirv.h
 * header already defines for Op* (which has shifted between spec
 * revisions).  If an identifier is missing, build errors surface
 * quickly — which is the safe behaviour. */
#ifndef SpvStorageClassInput
#  define SpvStorageClassInput                       1
#endif
#ifndef SpvStorageClassUniform
#  define SpvStorageClassUniform                     2
#endif
#ifndef SpvStorageClassPushConstant
#  define SpvStorageClassPushConstant                9
#endif
#ifndef SpvStorageClassStorageBuffer
#  define SpvStorageClassStorageBuffer              12
#endif
#ifndef SpvStorageClassUniformConstant
#  define SpvStorageClassUniformConstant             0
#endif

/* ── Dynamic SPIR-V word buffer ─────────────────────────────────────────── */
typedef struct {
    uint32_t *w;
    size_t n;
    size_t cap;
} SpvBuf;
static bool
sb_init(
    SpvBuf *b,
    size_t c)
{
    b->w = malloc(c * sizeof(uint32_t));
    b->n = 0;
    b->cap = c;
    return b->w != NULL;
}
static void
sb_free(
    SpvBuf *b)
{
    free(b->w);
    b->w = NULL;
    b->n = 0;
    b->cap = 0;
}
static bool
sb_push(
    SpvBuf *b,
    uint32_t v)
{
    if (b->n >= b->cap)
    {
        size_t new_cap = b->cap ? b->cap * 2 : 64;
        uint32_t *p = realloc(
            b->w,
            new_cap * sizeof(uint32_t));
        if (!p)
            return false;
        b->w = p;
        b->cap = new_cap;
    }
    b->w[b->n++] = v;
    return true;
}
static bool
sb_push_n(
    SpvBuf *b,
    const uint32_t *v,
    size_t c)
{
    for (size_t i = 0; i < c; i++)
    {
        if (!sb_push(b, v[i]))
            return false;
    }
    return true;
}
static inline uint32_t
op_(
    uint32_t op,
    uint32_t wc)
{
    return (wc << 16) | op;
}

/* ── Matrix provenance helpers ───────────────────────────────────────────── */

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
    uint32_t ut;
    uint32_t bt_type;
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
    /* Shader analysis */
    uint32_t dot_count;
    uint32_t fma_count;       /* GLSL.std.450 Fma (OpExtInst #50) count */
    bool has_matrix_ops;
    bool has_direct_position_write;
    bool has_geom_pos_store;
    uint32_t position_source_id;
    /* Matrix provenance tracking */
    uint32_t value_capacity;
    uint8_t *value_from_matrix;
    uint8_t *is_matrix_type;
    uint8_t *is_matrix_ptr;
    /* Projection UBO discovery */
    uint32_t proj_struct_type;
    uint32_t proj_ptr_type;
    uint32_t proj_var;
    uint32_t proj_set;
    uint32_t proj_binding;
    uint32_t proj_member_mask;
    VkBool32 proj_found;
    /* projection load tracking */
    uint32_t proj_access_count;
    uint32_t proj_load_count;
    uint32_t proj_mtv_count;
    /* Projection provenance per SSA value */
    uint8_t *is_proj_value;
    uint8_t *is_view_value;
    /* IO provenance per SSA value.
     * Set when a value directly or indirectly originates from
     *   SpvStorageClassInput (vertex attributes)
     *   SpvStorageClassUniform (constant buffer / UBO)
     *   SpvStorageClassPushConstant (push constants)
     * Fullscreen-quad / UI vertex shaders compute gl_Position
     * from compile-time constants only; any value reading real
     * geometry data must flow through Input or Uniform loads.
     * This is a more robust criterion than has_matrix_ops on
     * emulators (Yuzu/eden) where mat4*vec4 is decomposed into
     * scalar OpFMul/OpFAdd/OpFma without any OpTypeMatrix. */
    uint8_t *is_io_value;
    /* Parallel array for pointers (same as is_io_value but for
     * pointer-type SSA values; propagated through AccessChain /
     * CopyObject / Bitcast). */
    uint8_t *is_io_ptr;
    /* AccessChain base pointer lookup: for each SSA id that is the
     * result of an OpAccessChain, record its base pointer id.
     * 0 = not an AccessChain result. */
    uint32_t *ac_base;
    /* ---------- NEW ---------- */
    /* Value was reconstructed from depth/screen-space rather than world-space. */
    uint8_t *is_screen_value;
    /* Number of reconstruction operations detected. */
    uint32_t screen_reconstruct_count;
    /* Constant provenance per SSA value.
     * Set when a value is (transitively) derived from OpConstant
     * or OpConstantComposite only — i.e. a compile-time literal.
     * Used to detect 2D / fullscreen-quad shaders where gl_Position
     * is assembled from constants (z=0, w=1) rather than computed
     * from a projection matrix times geometry data. */
    uint8_t *is_const_value;
    /* ID of the value stored to gl_Position (tracked across
     * OpStore + AccessChain component writes).  0 = unknown. */
    uint32_t pos_stored_id;
    /* Whether gl_Position.w component was traced to a constant.
     * Set during scan by backtracking the w-operand of the
     * OpCompositeConstruct or OpStore that wrote gl_Position.w. */
    bool pos_w_is_const;
    /* True if any store to gl_Position (or .w component) was from
     * a non-const source.  When true, the final pos_w_is_const is
     * false (overriding the initializer pattern). */
    bool pos_w_seen_nonconst;
    /* AccessChain result id that targets gl_Position.w (member 3).
     * Used to detect per-component writes to .w.  0 = not seen. */
    uint32_t pos_w_accesschain;
    /* AccessChain result id that targets the entire gl_Position vec4
     * in block mode (OpAccessChain %block_var %int_0).  When a const-
     * derived vec4 is stored through it, the shader is a 2D/fullscreen
     * quad.  0 = not seen. */
    uint32_t pos_whole_accesschain;
    /* Per-value: tracks the .w component id of a vec4 CompositeConstruct.
     * Used to detect ScreenRectQuad pattern where gl_Position =
     * matrix * vec4(x, y, 0.0, 1.0) — the vec4's .w is a constant 1.0. */
    uint32_t *vec4_w_id;
    /* Per-value: true if this id is the result of OpMatrixTimesVector
     * where the vector operand had a constant .w component.  This is
     * the hallmark of 2D / fullscreen quad shaders. */
    uint8_t *mtv_vec_w_const;
} SpvMod;

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

static inline uint8_t proj_value(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->is_proj_value[id] : 0;
}

static inline void set_proj_value(SpvMod *m, uint32_t id, uint8_t value)
{
    if (valid_id(m, id))
        m->is_proj_value[id] = value;
}

static inline uint8_t view_value(const SpvMod *m, uint32_t id)
{
    return (id < m->value_capacity) ? m->is_view_value[id] : 0;
}

static inline void set_view_value(SpvMod *m, uint32_t id, uint8_t v)
{
    if (id < m->value_capacity)
        m->is_view_value[id] = v;
}

/* ── IO provenance helpers ──────────────────────────────────────────────── */
static inline uint8_t io_value(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->is_io_value[id] : 0;
}

static inline void set_io_value(SpvMod *m, uint32_t id, uint8_t v)
{
    if (valid_id(m, id))
        m->is_io_value[id] = v;
}

static inline uint8_t io_ptr(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->is_io_ptr[id] : 0;
}

static inline void set_io_ptr(SpvMod *m, uint32_t id, uint8_t v)
{
    if (valid_id(m, id))
        m->is_io_ptr[id] = v;
}

/* Constant provenance: a value is "const-derived" if it comes from
 * OpConstant, OpConstantComposite, OpConstantTrue/False, or any
 * operation whose operands are all const-derived. */
static inline uint8_t const_value(const SpvMod *m, uint32_t id)
{
    return valid_id(m, id) ? m->is_const_value[id] : 0;
}

static inline void set_const_value(SpvMod *m, uint32_t id, uint8_t v)
{
    if (valid_id(m, id))
        m->is_const_value[id] = v;
}

/* True if all non-type operands of an instruction are const-derived. */
static inline bool operands_all_const(
    const SpvMod *m,
    const uint32_t *w,
    size_t i,
    uint32_t wc,
    uint32_t first_operand_idx)
{
    for (uint32_t k = first_operand_idx; k < wc; ++k)
    {
        uint32_t id = w[i + k];
        if (id < m->value_capacity && !const_value(m, id))
            return false;
    }
    return true;
}

static inline uint8_t io_or2(const SpvMod *m, uint32_t a, uint32_t b)
{
    return io_value(m, a) | io_value(m, b);
}

static inline uint8_t io_or_multi(const SpvMod *m,
                                   const uint32_t *words,
                                   uint32_t start_k,
                                   uint32_t end_k_excl)
{
    uint8_t r = 0;
    for (uint32_t k = start_k; k < end_k_excl; ++k)
    {
        uint32_t id = words[k];
        if (id < m->value_capacity)
            r |= io_value(m, id);
    }
    return r;
}

static inline uint8_t matrix_or2(const SpvMod *m,
                                 uint32_t a,
                                 uint32_t b)
{
    return matrix_value(m, a) | matrix_value(m, b);
}

static void free_spv_provenance(SpvMod *m)
{
    free(m->value_from_matrix);
    free(m->is_matrix_type);
    free(m->is_matrix_ptr);
    free(m->is_proj_value);
    free(m->is_view_value);
    free(m->is_io_value);
    free(m->is_io_ptr);
    free(m->ac_base);
    free(m->is_const_value);
    free(m->vec4_w_id);
    free(m->mtv_vec_w_const);
    m->value_from_matrix = NULL;
    m->is_matrix_type    = NULL;
    m->is_matrix_ptr     = NULL;
    m->is_proj_value     = NULL;
    m->is_view_value     = NULL;
    m->is_io_value       = NULL;
    m->is_io_ptr         = NULL;
    m->ac_base           = NULL;
    m->is_const_value    = NULL;
    m->vec4_w_id         = NULL;
    m->mtv_vec_w_const   = NULL;
    m->value_capacity = 0;
}

static uint64_t hash_spv(const uint32_t *data, size_t words);

static bool
spv_resolve_u32_constant(const SpvMod *m, uint32_t id, uint32_t *value)
{
    if (!m || !value || !m->words)
        return false;
    for (size_t i = 5; i < m->count; )
    {
        uint32_t op = m->words[i] & 0xffffu;
        uint32_t wc = m->words[i] >> 16;
        if (!wc || i + wc > m->count)
            break;
        if (op == SpvOpConstant && wc >= 4 && m->words[i + 2] == id)
        {
            *value = m->words[i + 3];
            return true;
        }
        i += wc;
    }
    return false;
}

static const char *
spv_op_name(uint32_t op);

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
    #define PROJ(id)       proj_value(m, (id))
    #define SETPROJ(id,v)  set_proj_value(m, (id), (v))
    #define VIEW(id)       view_value(m, (id))
    #define SETVIEW(id,v)  set_view_value(m, (id), (v))
    #define CONSTV(id)     const_value(m, (id))
    #define SETCONST(id,v) set_const_value(m, (id), (v))
    for (size_t i=5;i<m->count;) {
        uint32_t op=w[i]&0xffff, wc=w[i]>>16;
        if (!wc||i+wc>m->count) break;
        if (!p2)
        {
        switch(op) {
            case SpvOpDot:
                m->dot_count++;
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    uint8_t proj = 0;
                    uint8_t view = 0;
                    uint8_t io = 0;
                    if (w[i + 3] < m->value_capacity)
                    {
                        matrix |= MAT(w[i + 3]);
                        io |= io_value(m, w[i + 3]);
                        if (PROJ(w[i + 3]))
                            proj = PROJ(w[i + 3]);
                        if (VIEW(w[i + 3]))
                            view = VIEW(w[i + 3]);
                    }
                    if (w[i + 4] < m->value_capacity)
                    {
                        matrix |= MAT(w[i + 4]);
                        io |= io_value(m, w[i + 4]);
                        if (!proj && PROJ(w[i + 4]))
                            proj = PROJ(w[i + 4]);
                        if (!view && VIEW(w[i + 4]))
                            view = VIEW(w[i + 4]);
                    }
                    SETMAT(w[i + 2], matrix);
                    set_io_value(m, w[i + 2], io);
                    if (proj)
                        SETPROJ(w[i + 2], proj);
                    if (view)
                        SETVIEW(w[i + 2], view);
                }
                break;
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
            case SpvOpPtrAccessChain:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity)
                {
                    /* Record base pointer for fast lookup in OpStore */
                    m->ac_base[w[i + 2]] = w[i + 3];
                    if (w[i + 3] < m->value_capacity)
                    {
                        SETPTR(w[i + 2], PTR(w[i + 3]));
                        set_io_ptr(m, w[i + 2], io_ptr(m, w[i + 3]));
                        if (io_ptr(m, w[i + 3]))
                        {
                            STEREO_LOG_VERBOSE(
                                "IO_ACCESS result=%u base=%u io_ptr=%u",
                                w[i + 2],
                                w[i + 3],
                                (unsigned)io_ptr(m, w[i + 3]));
                        }
                    }
                    /* Track per-component access to gl_Position.
                     * When OpStore writes through this pointer later,
                     * we use this member index to know which component
                     * (.x/.y/.z/.w) was written.  This is critical for
                     * detecting 2D shaders where .w is set to a constant
                     * 1.0 via per-component store. */
                    if (wc >= 5 && w[i + 3] == m->pos_var)
                    {
                        uint32_t member_idx = 0;
                        uint32_t const_lookup = w[i + 4];
                        uint32_t const_val = 0;
                        if (spv_resolve_u32_constant(
                                m, const_lookup, &const_val))
                            member_idx = const_val;
                        if (member_idx == 3)
                        {
                            /* Direct .w component access (non-block) */
                            m->pos_w_accesschain = w[i + 2];
                        }
                        else if (m->pos_is_block && member_idx == 0)
                        {
                            /* Block mode: %block_var %int_0 accesses the
                             * entire gl_Position vec4.  Track this as
                             * a "whole position" accesschain.  When a
                             * const-derived vec4 is stored through it,
                             * the shader is a 2D/fullscreen quad. */
                            m->pos_whole_accesschain = w[i + 2];
                        }
                    }
                }
                if (wc >= 5 &&
                    w[i+3] == m->proj_var)
                {
                    uint32_t member_id = w[i + 4];
                    uint32_t member_value = member_id;
                    (void)spv_resolve_u32_constant(
                        m,
                        member_id,
                        &member_value);
                    m->proj_access_count++;
                    m->proj_found = VK_TRUE;
                    /* Tag any member of the projection struct.
                     * The original code only accepted members 0-5
                     * (view/viewI/proj/projI/viewProj/prevViewProj),
                     * but Switch/Yuzu uses a different UBO layout.
                     * Tag all members; the patching code will only
                     * apply offset to values used in matrix ops. */
                    SETPROJ(
                        w[i + 2],
                        member_value + 1);
                    /* If member is 2 (projection), also tag as view */
                    if (member_value == 2)
                        SETVIEW(
                            w[i + 2],
                            1);
                    STEREO_LOG(
                        "PROJ_ACCESS_HIT result=%u base=%u member=%u "
                        "count=%u set=%u binding=%u",
                        w[i+2],
                        w[i+3],
                        member_value,
                        m->proj_access_count,
                        m->proj_set,
                        m->proj_binding);
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
                    /* Propagate IO provenance from pointer.
                     * If the pointer is itself an IO pointer, or the
                     * pointer object it names is an IO pointer, the
                     * loaded value derives from Input / Uniform. */
                    uint8_t io_src =
                        io_ptr(m, w[i + 3]) || io_value(m, w[i + 3]);
                    set_io_value(m, w[i + 2], io_src);
                    if (io_src)
                    {
                        STEREO_LOG_VERBOSE(
                            "IO_LOAD result=%u ptr=%u io_ptr=%u io_val=%u",
                            w[i + 2],
                            w[i + 3],
                            (unsigned)io_ptr(m, w[i + 3]),
                            (unsigned)io_value(m, w[i + 3]));
                    }
                }
                if (wc >= 4)
                {
                    if (PROJ(w[i + 3]))
                    {
                        SETPROJ(w[i + 2], PROJ(w[i + 3]));
                        m->proj_load_count++;
                        STEREO_LOG_VERBOSE(
                            "PROJ_LOAD id=%u src=%u count=%u proj=%u",
                            w[i+2],
                            w[i+3],
                            m->proj_load_count,
                            PROJ(w[i + 2]));
                    }
                    if (VIEW(w[i + 3]))
                    {
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
                    }
                }
                break;
            case SpvOpCompositeExtract:
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                    set_io_value(m, w[i + 2], io_value(m, w[i + 3]));
                    if (PROJ(w[i + 3]))
                    {
                        STEREO_LOG_VERBOSE(
                            "PROJ_EXTRACT result=%u src=%u member=%u",
                            w[i + 2],
                            w[i + 3],
                            PROJ(w[i + 3]) - 1);
                        SETPROJ(w[i + 2], PROJ(w[i + 3]));
                    }
                    if (VIEW(w[i + 3]))
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
                    if (VIEW(w[i + 3]) != 0)
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
                }
                break;
            case SpvOpVectorShuffle:
                if (wc >= 6 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                    set_io_value(
                        m,
                        w[i + 2],
                        io_or2(m, w[i + 3], w[i + 4]));
                    if (PROJ(w[i + 3]))
                        SETPROJ(w[i + 2], PROJ(w[i + 3]));
                    if (VIEW(w[i + 3]))
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
                }
                break;
            case SpvOpCompositeConstruct:
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    uint8_t proj = 0;
                    uint8_t view = 0;
                    uint8_t io = 0;
                    uint8_t all_const = 1;
                    for (uint32_t k = 3; k < wc; ++k)
                    {
                        uint32_t id = w[i + k];
                        if (id >= m->value_capacity)
                        {
                            all_const = 0;
                            continue;
                        }
                        matrix |= MAT(id);
                        io |= io_value(m, id);
                        if (!proj && PROJ(id))
                            proj = PROJ(id);
                        if (!view && VIEW(id))
                            view = VIEW(id);
                        if (!CONSTV(id))
                            all_const = 0;
                    }
                    SETMAT(w[i + 2], matrix);
                    set_io_value(m, w[i + 2], io);
                    SETCONST(w[i + 2], all_const);
                    if (proj)
                        SETPROJ(w[i + 2], proj);
                    if (view)
                        SETVIEW(w[i + 2], view);
                    /* Track .w component id for vec4 CompositeConstruct.
                     * Layout: CompositeConstruct %v4float %x %y %z %w
                     * If wc == 7, w operand is at w[i + 6].  This lets
                     * us detect ScreenRectQuad pattern where gl_Position
                     * = matrix * vec4(x, y, 0.0, 1.0) — the .w operand
                     * is a constant 1.0. */
                    if (wc == 7 && w[i + 6] < m->value_capacity)
                        m->vec4_w_id[w[i + 2]] = w[i + 6];
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
                if (wc == 4 && w[i + 2] == 32)
                {
                    if (w[i + 3] == 1)
                    {
                        m->it = w[i + 1];
                    }
                    else
                    {
                        m->ut = w[i + 1];
                    }
                }
                break;
            case SpvOpTypeBool:
                if (wc >= 2 && !m->bt_type)
                    m->bt_type = w[i + 1];
                break;
            case SpvOpTypeMatrix:
                if (wc >= 4)
                {
                    if (w[i + 1] < m->value_capacity)
                        SETTYPE(w[i + 1], 1);
                    /* Log matrix dimensions for projection detection.
                     * True projection matrix is 4x4 float. 3x3 = normal
                     * matrix, 4x3 = world, etc. */
                    if (w[i + 2] < m->value_capacity &&
                        TYPE(w[i + 2]) &&
                        w[i + 2] == m->ft)
                    {
                        STEREO_LOG(
                            "PROJ_MATRIX_TYPE id=%u cols=%u rows_type=%u "
                            "(4x4 float = projection candidate)",
                            w[i + 1],
                            w[i + 3],
                            w[i + 2]);
                    }
                    else
                    {
                        STEREO_LOG_VERBOSE(
                            "MATRIX_TYPE id=%u cols=%u row_type=%u "
                            "(non-float or non-4-rows)",
                            w[i + 1],
                            w[i + 3],
                            w[i + 2]);
                    }
                }
                break;
            case SpvOpTypeStruct:
                if (wc >= 3)
                {
                    uint8_t matrix = 0;
                    uint32_t matrix_member_idx = 0;
                    for (uint32_t k = 2; k < wc; ++k)
                    {
                        if (w[i + k] < m->value_capacity &&
                            TYPE(w[i + k]))
                        {
                            matrix = 1;
                            matrix_member_idx = k - 2;
                            break;
                        }
                    }
                    if (w[i + 1] < m->value_capacity)
                        SETTYPE(w[i + 1], matrix);
                    if (matrix)
                    {
                        STEREO_LOG(
                            "PROJ_STRUCT_TYPE type=%u members=%u "
                            "matrix_member=%u (previous=%u)",
                            w[i+1],
                            wc - 2,
                            matrix_member_idx,
                            m->proj_struct_type);
                        m->proj_struct_type = w[i+1];
                    }
                    else
                    {
                        STEREO_LOG_VERBOSE(
                            "STRUCT_NO_MATRIX type=%u members=%u",
                            w[i+1],
                            wc - 2);
                    }
                }
                break;
            case SpvOpTypeArray:
                if (wc >= 4)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_TYPE_ARRAY id=%u elem=%u len=%u",
                        w[i + 1],
                        w[i + 2],
                        w[i + 3]);
                }
                break;
            case SpvOpTypeRuntimeArray:
                break;
            /* Constant provenance: mark all OpConstant* results as
             * const-derived.  This is the seed for propagating
             * const-ness through arithmetic / composite ops. */
            case SpvOpConstantTrue:
            case SpvOpConstantFalse:
            case SpvOpConstant:
            case SpvOpConstantNull:
                if (wc >= 3 && w[i + 2] < m->value_capacity)
                    SETCONST(w[i + 2], 1);
                break;
            case SpvOpConstantComposite:
                /* Composite of constants is constant.  If any
                 * constituent is non-const, mark as non-const. */
                if (wc >= 3 && w[i + 2] < m->value_capacity)
                {
                    bool all_const = true;
                    for (uint32_t k = 3; k < wc; ++k)
                    {
                        if (w[i + k] >= m->value_capacity) continue;
                        if (!CONSTV(w[i + k]))
                        {
                            all_const = false;
                            break;
                        }
                    }
                    SETCONST(w[i + 2], all_const ? 1 : 0);
                }
                break;
            case SpvOpTranspose:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                    if (PROJ(w[i + 3]))
                        SETPROJ(w[i + 2], PROJ(w[i + 3]));
                    if (VIEW(w[i + 3]))
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
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
                        uint32_t a = w[i + 3];
                        uint32_t b = w[i + 4];
                        uint8_t proj_a = PROJ(a);
                        uint8_t proj_b = PROJ(b);
                        uint8_t view_a = VIEW(a);
                        uint8_t view_b = VIEW(b);
                        if (op == SpvOpMatrixTimesVector || op == SpvOpMatrixTimesMatrix)
                        {
                            STEREO_LOG_VERBOSE(
                                "FS_MATRIX_MUL op=%s result=%u a=%u b=%u proj_a=%u proj_b=%u view_a=%u view_b=%u",
                                spv_op_name(op),
                                w[i + 2],
                                a,
                                b,
                                proj_a,
                                proj_b,
                                view_a,
                                view_b);
                            /* Detect ScreenRectQuad pattern:
                             * gl_Position = matrix * vec4(x, y, 0.0, 1.0)
                             * The vector operand (b) is a CompositeConstruct
                             * whose .w component (4th operand) is a constant.
                             * This means gl_Position.w will be constant 1.0
                             * for ALL vertices → 2D/fullscreen quad, NOT 3D. */
                            if (op == SpvOpMatrixTimesVector &&
                                b < m->value_capacity &&
                                m->vec4_w_id[b])
                            {
                                uint32_t w_id = m->vec4_w_id[b];
                                if (w_id < m->value_capacity &&
                                    const_value(m, w_id))
                                {
                                    m->mtv_vec_w_const[w[i + 2]] = 1;
                                    STEREO_LOG(
                                        "MTV_VEC_W_CONST result=%u vec=%u "
                                        "w_id=%u (ScreenRectQuad: vector .w "
                                        "is constant → 2D shader)",
                                        w[i + 2],
                                        b,
                                        w_id);
                                }
                            }
                        }
                        if ((proj_a || proj_b) &&
                            (op == SpvOpMatrixTimesVector ||
                             op == SpvOpMatrixTimesMatrix))
                        {
                            uint8_t proj = proj_a ? proj_a : proj_b;
                            m->proj_member_mask |= 1u << (proj - 1);
                            m->proj_mtv_count++;
                            STEREO_LOG_VERBOSE(
                                "PROJ_MTV result=%u matrix=%u vector=%u member=%u mask=0x%X count=%u",
                                w[i + 2],
                                a,
                                b,
                                proj - 1,
                                m->proj_member_mask,
                                m->proj_mtv_count);
                        }
                        /* Do not automatically propagate projection provenance through
                         * MatrixTimesVector.
                         *
                         * Many fragment shaders (SSAO, SSR, depth reconstruction) multiply
                         * arbitrary vectors by the projection matrix without producing clip
                         * coordinates.
                         *
                         * Let later consumers decide whether this multiplication is actually
                         * part of a projection chain.
                         */
                        //if (proj_a || proj_b)
                        //    SETPROJ(w[i + 2], proj_a ? proj_a : proj_b);
                        if (view_a || view_b)
                            SETVIEW(w[i + 2], view_a ? view_a : view_b);
                        SETMAT(w[i + 2], 1);
                        set_io_value(
                            m,
                            w[i + 2],
                            io_or2(m, a, b));
                    }
                }
                break;
            case SpvOpCopyObject:
            case SpvOpBitcast:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    w[i + 3] < m->value_capacity)
                {
                    SETMAT(w[i + 2], MAT(w[i + 3]));
                    set_io_value(m, w[i + 2], io_value(m, w[i + 3]));
                    set_io_ptr(m, w[i + 2], io_ptr(m, w[i + 3]));
                    if (PROJ(w[i + 3]))
                        SETPROJ(w[i + 2], PROJ(w[i + 3]));
                    if (VIEW(w[i + 3]))
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
                }
                break;
            case SpvOpExtInst:
                /* Detect GLSL.std.450 Fma (instruction #50).
                 * Eden/Yuzu decomposes mat4*vec4 into FMul + Fma chains:
                 *   result = a*b + c  (fused multiply-add)
                 * A full 4x4 matrix * vec4 requires 12 Fma ops (3 per row).
                 * This is the hallmark of decomposed matrix multiplication
                 * that does NOT use OpTypeMatrix/OpMatrixTimesVector. */
                if (wc >= 5 && w[i + 4] == 50)
                    m->fma_count++;
                if (wc >= 7 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    uint8_t proj = 0;
                    uint8_t view = 0;
                    uint8_t io = 0;
                    uint8_t all_const = 1;
                    for (uint32_t k = 5; k < wc; ++k)
                    {
                        uint32_t id = w[i + k];
                        if (id >= m->value_capacity)
                        {
                            all_const = 0;
                            continue;
                        }
                        matrix |= MAT(id);
                        io |= io_value(m, id);
                        if (!proj && PROJ(id))
                            proj = PROJ(id);
                        if (!view && VIEW(id))
                            view = VIEW(id);
                        if (!CONSTV(id))
                            all_const = 0;
                    }
                    SETMAT(w[i + 2], matrix);
                    set_io_value(m, w[i + 2], io);
                    SETCONST(w[i + 2], all_const);
                    if (proj)
                        SETPROJ(w[i + 2], proj);
                    if (view)
                        SETVIEW(w[i + 2], view);
                }
                break;
            case SpvOpFAdd:
            case SpvOpFSub:
            case SpvOpFMul:
            case SpvOpFDiv:
                /* SPIR-V layout: opcode|wc, ResultType, Result,
                 * Operand1, Operand2  →  operands at w[i+3], w[i+4] */
                if (wc >= 5 &&
                    w[i + 2] < m->value_capacity)
                {
                    SETMAT(
                        w[i + 2],
                        matrix_or2(m, w[i + 3], w[i + 4]));
                    set_io_value(
                        m,
                        w[i + 2],
                        io_or2(m, w[i + 3], w[i + 4]));
                    /* const propagation: result is const iff both
                     * operands are const. */
                    SETCONST(
                        w[i + 2],
                        (CONSTV(w[i + 3]) && CONSTV(w[i + 4])) ? 1 : 0);
                    if (PROJ(w[i + 3]))
                        SETPROJ(w[i + 2], PROJ(w[i + 3]));
                    else if (PROJ(w[i + 4]))
                        SETPROJ(w[i + 2], PROJ(w[i + 4]));
                    if (VIEW(w[i + 3]))
                        SETVIEW(w[i + 2], VIEW(w[i + 3]));
                    else if (VIEW(w[i + 4]))
                        SETVIEW(w[i + 2], VIEW(w[i + 4]));
                }
                break;
            /* Note: additional arithmetic operations intentionally not
             * enumerated individually.  MAT and IO provenance are still
             * propagated correctly because matrix chains on Yuzu use
             * either OpDot (handled above) or OpFMul+OpFAdd (handled
             * above), and any intermediate integer/index arithmetic
             * ultimately feeds an OpAccessChain whose pointer is
             * already tagged as IO via the SpvStorageClassInput/Uniform
             * variable marking. */
            case SpvOpSelect:
                if (wc >= 6 &&
                    w[i + 2] < m->value_capacity)
                {
                    SETMAT(
                        w[i + 2],
                        matrix_or2(m, w[i + 4], w[i + 5]));
                    set_io_value(
                        m,
                        w[i + 2],
                        io_or2(m, w[i + 4], w[i + 5]));
                    if (PROJ(w[i + 4]))
                        SETPROJ(w[i + 2], PROJ(w[i + 4]));
                    else if (PROJ(w[i + 5]))
                        SETPROJ(w[i + 2], PROJ(w[i + 5]));
                    if (VIEW(w[i + 4]))
                        SETVIEW(w[i + 2], VIEW(w[i + 4]));
                    else if (VIEW(w[i + 5]))
                        SETVIEW(w[i + 2], VIEW(w[i + 5]));
                }
                break;
            case SpvOpFunctionCall:
                break;
            case SpvOpCompositeInsert:
                if (wc >= 6 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t matrix = 0;
                    uint8_t proj = 0;
                    uint8_t view = 0;
                    uint8_t io = 0;
                    if (w[i + 3] < m->value_capacity)
                    {
                        matrix |= MAT(w[i + 3]);
                        io |= io_value(m, w[i + 3]);
                        if (PROJ(w[i + 3]))
                            proj = PROJ(w[i + 3]);
                        if (VIEW(w[i + 3]))
                            view = VIEW(w[i + 3]);
                    }
                    if (w[i + 4] < m->value_capacity)
                    {
                        matrix |= MAT(w[i + 4]);
                        io |= io_value(m, w[i + 4]);
                        if (!proj && PROJ(w[i + 4]))
                            proj = PROJ(w[i + 4]);
                        if (!view && VIEW(w[i + 4]))
                            view = VIEW(w[i + 4]);
                    }
                    SETMAT(w[i + 2], matrix);
                    set_io_value(m, w[i + 2], io);
                    if (proj)
                        SETPROJ(w[i + 2], proj);
                    if (view)
                        SETVIEW(w[i + 2], view);
                }
                break;
            case SpvOpTypePointer:
                if (wc >= 4)
                {
                if (TYPE(w[i + 3]))
                {
                SETPTR(w[i + 1], 1);
                }
                if (w[i+3] == m->proj_struct_type)
                {
                    STEREO_LOG_VERBOSE(
                        "PROJ_PTR ptr=%u struct=%u",
                        w[i+1],
                        w[i+3]);
                    m->proj_ptr_type = w[i+1];
                }
                if (w[i + 2] == SpvStorageOutput &&
                m->v4t &&
                w[i + 3] == m->v4t)
                {
                m->ptr_out_v4 = w[i + 1];
                }
                if (w[i + 2] == SpvStorageInput)
                {
                    STEREO_LOG_VERBOSE(
                        "VS_INPUT_POINTER ptr=%u pointeeType=%u",
                        w[i + 1],
                        w[i + 3]);
                    if (m->ut &&
                        w[i + 3] == m->ut)
                    {
                        STEREO_LOG_VERBOSE(
                            "VS_UINT_POINTER ptr=%u",
                            w[i + 1]);
                        m->ptr_in_int = w[i + 1];
                    }
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
                /* IO provenance: mark pointers that originate from
                 * any shader-input storage class as IO pointers.
                 * This covers all ways Yuzu/eden may pass geometry
                 * data to a VS without traditional vertex bindings:
                 *
                 *   UniformConstant  (0)  — textures, atomic counters
                 *   Input            (1)  — vertex attributes
                 *   Uniform          (2)  — UBO / cbuf (Switch Maxwell)
                 *   PushConstant     (9)  — push constants
                 *   StorageBuffer   (12)  — SSBO (GPU-driven vertex fetch)
                 *
                 * Any OpLoad from such a pointer (or derivative via
                 * AccessChain) will propagate the IO tag to the
                 * loaded SSA value.  Mathematics preserves the tag.
                 * This lets us reliably distinguish fullscreen-quad
                 * shaders (gl_Position built from constants only)
                 * from real geometry shaders (gl_Position consumes
                 * vertex attributes, UBO, SSBO, or push constants). */
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity &&
                    (w[i + 3] == SpvStorageClassUniformConstant ||
                     w[i + 3] == SpvStorageInput ||
                     w[i + 3] == SpvStorageClassUniform ||
                     w[i + 3] == SpvStorageClassPushConstant ||
                     w[i + 3] == SpvStorageClassStorageBuffer))
                {
                    set_io_ptr(m, w[i + 2], 1);
                    STEREO_LOG_VERBOSE(
                        "IO_VAR_MARKED var=%u storage=%u io_ptr=1",
                        w[i + 2],
                        w[i + 3]);
                }
                if (w[i+1] == m->proj_ptr_type &&
                    (w[i+3] == SpvStorageClassUniform ||
                     w[i+3] == SpvStorageClassPushConstant))
                {
                    STEREO_LOG(
                        "PROJ_VAR_CANDIDATE var=%u ptr=%u storage=%u(%s) "
                        "previous=%u",
                        w[i+2],
                        w[i+1],
                        w[i+3],
                        w[i+3] == SpvStorageClassUniform ?
                            "Uniform(UBO)" : "PushConstant",
                        m->proj_var);
                    m->proj_var = w[i+2];
                }
                else if (w[i+1] == m->proj_ptr_type)
                {
                    STEREO_LOG_VERBOSE(
                        "PROJ_VAR_SKIPPED var=%u ptr=%u storage=%u "
                        "(not Uniform/PushConstant)",
                        w[i+2],
                        w[i+1],
                        w[i+3]);
                }
                if (w[i + 3] == SpvStorageInput)
                {
                    STEREO_LOG_VERBOSE(
                        "VS_INPUT_VARIABLE var=%u ptr=%u",
                        w[i + 2],
                        w[i + 1]);
                }
                break;
            case SpvOpDecorate:
                if (wc >= 4)
                {
                    if (w[i+2] == SpvDecorationDescriptorSet &&
                        w[i+1] == m->proj_var)
                    {
                        m->proj_set = w[i+3];
                        STEREO_LOG(
                            "PROJ_SET_FOUND var=%u set=%u",
                            w[i+1],
                            w[i+3]);
                    }
                    if (w[i+2] == SpvDecorationBinding &&
                        w[i+1] == m->proj_var)
                    {
                        m->proj_binding = w[i+3];
                        STEREO_LOG(
                            "PROJ_BINDING_FOUND var=%u binding=%u",
                            w[i+1],
                            w[i+3]);
                    }
                }
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
                if (wc >= 3)
                {
                    uint32_t ptr_id = w[i + 1];
                    uint32_t source = w[i + 2];
                    bool is_pos_store = false;

                    if (ptr_id == m->pos_var)
                    {
                        is_pos_store = true;
                        if (current_function && !m->position_function)
                            m->position_function = current_function;
                        m->position_source_id = source;
                    }
                    else
                    {
                        /* Walk up AccessChain chain using pre-built
                         * ac_base[] lookup table.  This catches
                         * per-component stores like:
                         *   %ac = OpAccessChain OutputVecPtr %pos %idx
                         *   OpStore %ac %computed_component */
                        uint32_t cur = ptr_id;
                        for (uint32_t depth = 0; depth < 32 && cur != 0; ++depth)
                        {
                            if (cur == m->pos_var)
                            {
                                is_pos_store = true;
                                if (current_function && !m->position_function)
                                    m->position_function = current_function;
                                m->position_source_id = source;
                                break;
                            }
                            if (cur >= m->value_capacity)
                                break;
                            uint32_t base = m->ac_base[cur];
                            if (base == 0)
                                break;
                            cur = base;
                        }
                    }

                    if (is_pos_store)
                    {
                        bool direct = (source >= m->value_capacity) ||
                                      (!MAT(source) && !io_value(m, source));

                        if (direct)
                            m->has_direct_position_write = true;
                        else
                            m->has_geom_pos_store = true;

                        /* Track gl_Position source for 2D filter.
                         * Many 3D shaders initialize gl_Position with
                         * a constant vec4(0,0,0,1) BEFORE overwriting
                         * each component via per-component stores.
                         * We must NOT permanently mark pos_w_is_const
                         * based on this initializer — subsequent
                         * per-component stores will overwrite .w with
                         * computed values.  Use a local "last store
                         * was const" tracking; final decision is made
                         * after scan completes based on whether ANY
                         * non-const store targeted gl_Position. */
                        m->pos_stored_id = source;
                        bool this_store_const =
                            (source < m->value_capacity) &&
                            const_value(m, source);
                        if (this_store_const)
                        {
                            /* Only mark as const if we haven't seen
                             * a non-const store yet.  Initializer
                             * pattern sets it true; real data stores
                             * will clear it below. */
                            if (!m->pos_w_seen_nonconst)
                                m->pos_w_is_const = true;
                        }
                        else
                        {
                            /* Non-const store: gl_Position gets real
                             * geometry/matrix data → not a 2D shader. */
                            m->pos_w_is_const = false;
                            m->pos_w_seen_nonconst = true;
                        }

                        STEREO_LOG(
                            "POS_STORE hash=%016llx ptr=%u direct=%u "
                            "source=%u io=%u mat=%u geom=%u dir=%u "
                            "const=%u w_const=%u",
                            (unsigned long long)hash_spv(m->words, m->count),
                            ptr_id,
                            (unsigned)direct,
                            source,
                            source < m->value_capacity ? (unsigned)io_value(m, source) : 0u,
                            source < m->value_capacity ? (unsigned)MAT(source) : 0u,
                            (unsigned)m->has_geom_pos_store,
                            (unsigned)m->has_direct_position_write,
                            source < m->value_capacity ? (unsigned)const_value(m, source) : 0u,
                            (unsigned)m->pos_w_is_const);
                    }
                    /* Per-component write to gl_Position.w via
                     * AccessChain %pos %int_3.  Track if the stored
                     * value is const-derived. */
                    if (ptr_id == m->pos_w_accesschain &&
                        source < m->value_capacity)
                    {
                        bool w_store_const = const_value(m, source);
                        if (w_store_const)
                        {
                            if (!m->pos_w_seen_nonconst)
                                m->pos_w_is_const = true;
                        }
                        else
                        {
                            m->pos_w_is_const = false;
                            m->pos_w_seen_nonconst = true;
                            STEREO_LOG(
                                "POS_W_DATA_STORE hash=%016llx source=%u "
                                "(gl_Position.w written from real data)",
                                (unsigned long long)hash_spv(m->words, m->count),
                                source);
                        }
                    }
                    /* Block mode: write to entire gl_Position vec4 via
                     * OpAccessChain %block_var %int_0.  Track const-ness
                     * of the source — a const vec4 (e.g. vec4(x,y,0,1)
                     * from constants) means a 2D/fullscreen quad. */
                    if (ptr_id == m->pos_whole_accesschain &&
                        source < m->value_capacity)
                    {
                        bool whole_store_const = const_value(m, source);
                        if (whole_store_const)
                        {
                            if (!m->pos_w_seen_nonconst)
                                m->pos_w_is_const = true;
                            STEREO_LOG(
                                "POS_WHOLE_CONST_STORE hash=%016llx source=%u "
                                "(gl_Position vec4 written from constants)",
                                (unsigned long long)hash_spv(m->words, m->count),
                                source);
                        }
                        else
                        {
                            m->pos_w_is_const = false;
                            m->pos_w_seen_nonconst = true;
                            STEREO_LOG(
                                "POS_WHOLE_DATA_STORE hash=%016llx source=%u "
                                "(gl_Position vec4 written from real data)",
                                (unsigned long long)hash_spv(m->words, m->count),
                                source);
                        }
                    }
                }
                break;
            /* ── Generic IO provenance fallback ───────────────────────
             * Catch any result-bearing instruction not handled above.
             * If any operand is tagged IO, the result inherits the tag.
             * This guarantees no propagation gap for opcodes we forgot
             * to enumerate (OpConvert*, OpFNegate, OpAtomic*, etc).
             * For type-decl / decoration / constant instructions the
             * "operands" may be plain integers — io_value() returns 0
             * for out-of-range ids, so stray integers are harmless. */
            default:
                if (wc >= 4 &&
                    w[i + 2] < m->value_capacity)
                {
                    uint8_t io = 0;
                    for (uint32_t k = 3; k < wc; ++k)
                    {
                        if (w[i + k] < m->value_capacity)
                            io |= io_value(m, w[i + k]);
                    }
                    if (io)
                        set_io_value(m, w[i + 2], io);
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
            /* Third pass (p2=true): now that pos_var is known,
             * detect AccessChain patterns that target gl_Position.
             * This must run AFTER pos_var is set (in this same pass,
             * earlier iterations).  Since SPIR-V orders OpVariable
             * before function bodies, pos_var is already set when
             * we reach OpAccessChain inside OpFunction. */
            if((op==SpvOpAccessChain||
                op==SpvOpInBoundsAccessChain||
                op==SpvOpPtrAccessChain) &&
               wc>=5 && w[i+3]==m->pos_var)
            {
                uint32_t const_lookup = w[i + 4];
                uint32_t const_val = 0;
                if (spv_resolve_u32_constant(m, const_lookup, &const_val))
                {
                    if (const_val == 3)
                    {
                        m->pos_w_accesschain = w[i + 2];
                    }
                    else if (m->pos_is_block && const_val == 0)
                    {
                        m->pos_whole_accesschain = w[i + 2];
                        STEREO_LOG(
                            "POS_WHOLE_AC hash=%016llx ac=%u base=%u "
                            "(block mode, member 0 = gl_Position vec4)",
                            (unsigned long long)hash_spv(m->words, m->count),
                            w[i + 2],
                            w[i + 3]);
                    }
                }
            }
            /* Also handle OpStore in third pass for block mode.
             * This catches stores to pos_whole_accesschain which
             * were not detected in first pass (pos_var was 0).
             * OpStore layout: w[i+1]=pointer, w[i+2]=value/source. */
            if(op==SpvOpStore && wc>=3 &&
               w[i+1]==m->pos_whole_accesschain)
            {
                uint32_t source = w[i + 2];
                if (source < m->value_capacity)
                {
                    bool whole_store_const = const_value(m, source);
                    /* Also check mtv_vec_w_const: the source is the
                     * result of OpMatrixTimesVector where the vector
                     * operand had a constant .w component.  This is
                     * the ScreenRectQuad pattern: gl_Position =
                     * matrix * vec4(x, y, 0.0, 1.0).  Even though
                     * the matrix*vec4 result is not "const" (matrix
                     * is runtime data), gl_Position.w is effectively
                     * constant 1.0 for all vertices. */
                    bool mtv_w_const = (source < m->value_capacity) &&
                                       m->mtv_vec_w_const[source];
                    if (whole_store_const || mtv_w_const)
                    {
                        if (!m->pos_w_seen_nonconst)
                        {
                            m->pos_w_is_const = true;
                            STEREO_LOG(
                                "POS_WHOLE_CONST_STORE hash=%016llx source=%u "
                                "%s(gl_Position vec4 written from %s)",
                                (unsigned long long)hash_spv(m->words, m->count),
                                source,
                                mtv_w_const ? "(mtv_vec_w_const) " : "",
                                mtv_w_const ? "matrix*vec4(x,y,0,1)" :
                                               "constants");
                        }
                    }
                    else
                    {
                        m->pos_w_is_const = false;
                        m->pos_w_seen_nonconst = true;
                    }
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
    /* ── 2D / Fullscreen-quad guard ─────────────────────────────────
     * BEFORE accepting has_matrix_ops / dot / fma as proof of a 3D
     * projection shader, reject shaders where gl_Position is written
     * from compile-time constants only.  Such shaders produce a flat
     * 2D rectangle (z=0, w=1) and applying OFF_AXIS parallax to them
     * causes uniform whole-screen shift, not depth-dependent parallax.
     *
     * Detection: the value stored to gl_Position (or its .w component)
     * is const-derived, meaning it ultimately comes from OpConstant
     * literals only — no IO (vertex attributes / UBO / push constants)
     * data flows into it.  True 3D shaders compute gl_Position.w from
     * a projection matrix * vertex data, so .w varies per vertex. */
    if (m->pos_w_is_const)
    {
        STEREO_LOG(
            "2D_SHADER_REJECTED hash=%016llx reason=POS_W_IS_CONST "
            "(gl_Position.w from constants only, 2D/fullscreen quad, "
            "proj_found reset to false, has_matrix_ops reset to false)",
            (unsigned long long)hash_spv(m->words, m->count));
        m->has_matrix_ops = false;
        m->proj_found = false;
    }
    /* Yuzu/Maxwell shaders decompose mat4 * vec4 into 4 OpDot products
     * plus OpCompositeConstruct (or per-component writes).  A trivial
     * passthrough or fullscreen UI shader generally has <4 dot ops and
     * may lack a position output entirely.  When dot_count >= 4 and a
     * position variable exists, treat the shader as if it contained
     * explicit OpMatrixTimesVector operations.  This avoids the
     * NO_MATRIX_OPS false-negative skip on Switch/Yuzu geometry. */
    if (!m->has_matrix_ops && m->dot_count >= 4 && m->pos_var)
    {
        m->has_matrix_ops = true;
    }
    /* Eden/Yuzu decomposes mat4*vec4 into FMul + GLSL.std.450 Fma chains
     * (OpExtInst #50).  A full 4x4 matrix * vec4 requires exactly 12 Fma
     * operations (3 remaining components per row × 4 rows).  When we see
     * >= 12 Fma ops and a position variable exists, the shader is doing
     * decomposed matrix multiplication — the 3D scene vertex shader.
     * This catches the pattern that OpTypeMatrix/OpMatrixTimesVector
     * detection misses entirely on Switch emulators. */
    if (!m->has_matrix_ops && m->fma_count >= 12 && m->pos_var)
    {
        m->has_matrix_ops = true;
        STEREO_LOG(
            "FMA_MATRIX_DETECTED hash=%016llx fma_count=%u "
            "(decomposed mat4*vec4 via GLSL.std.450 Fma)",
            (unsigned long long)hash_spv(m->words, m->count),
            m->fma_count);
    }
    STEREO_LOG_VERBOSE(
        "SCAN hash=%016llx exec=%u matrix=%u proj=%u dot=%u fma=%u direct=%u emit=%u pos=%u block=%u io_pos=%u",
        (unsigned long long)hash_spv(m->words, m->count),
        m->exec_model,
        m->has_matrix_ops,
        m->proj_found,
        m->dot_count,
        m->fma_count,
        m->has_direct_position_write,
        m->emit_count,
        m->pos_var,
        m->pos_is_block,
        (unsigned)m->pos_var && m->pos_var < m->value_capacity
            ? (unsigned)io_value(m, m->pos_var)
            : 0u);
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
        CHECK_ARRAY_COUNT(sd->cb_track_count, MAX_CB_TRACK, "cb_track_count");
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
    CHECK_ARRAY_COUNT(sd->cb_track_count, MAX_CB_TRACK, "cb_track_count");
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
    CHECK_ARRAY_COUNT(sd->pipeline_info_count, sd->pipeline_info_capacity, "pipeline_info_count");
    StereoPipelineInfo *info =
        &sd->pipeline_info[sd->pipeline_info_count++];
    memset(info, 0, sizeof(*info));
    return info;
}

/* ── Stereo offset injection body ────────────────────────────────────────── */
typedef struct {
    SpvMod *m;
    bool have_view;
    /* Provenance */
    bool has_projection_path;
    bool has_view_path;
    uint32_t uv4;
    uint32_t uint_;
    uint32_t ut;
    uint32_t bt;
    uint32_t cz;
    uint32_t cf0;
    uint32_t cl;
    uint32_t cr;
    uint32_t cc;
    uint32_t projection_mode;
    float lo_dbg;
    float ro_dbg;
    float conv_dbg;
    StereoDebugCtx *dbg;
} BodyCtx;

typedef struct StereoDebugCtx {
    uint32_t pipeline_index;
    VkRenderPass render_pass;
    bool is_multiview;
    uint32_t stage;
    uint32_t vertex_binding_count;
    uint32_t is_quad;
    VkBool32 has_proj_ubo;
    uint32_t proj_set;
    uint32_t proj_binding;
    uint32_t proj_member_mask;
    uint32_t proj_var;
    bool has_matrix_ops;
    bool direct_position_write;
} StereoDebugCtx;

static void emit_body(SpvBuf *out, const BodyCtx *c, uint32_t *nid)
{
    SpvMod *m = c->m;
    STEREO_LOG(
        "EMIT_BODY_START have_view=%d mode=%d cl=%u cr=%u cc=%u lo=%f ro=%f conv=%f",
        (int)c->have_view,
        c->projection_mode,
        c->cl,
        c->cr,
        c->cc,
        c->lo_dbg,
        c->ro_dbg,
        c->conv_dbg);
    STEREO_LOG(
        "EMIT_BODY_POS pos_var=%u pos_is_block=%u view_var=%u ut=%u",
        m->pos_var,
        m->pos_is_block ? 1 : 0,
        m->view_var,
        m->ut);
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
    /* VKS3D_DEBUG_VS_VIEW: when set, replace the sep/conv offset with a
     * direct ViewIndex*0.5 horizontal shift. This makes it visually
     * obvious whether gl_ViewIndex is actually populated per-view in
     * the VS stage:
     *   - If left/right halves show different horizontal shift → VS
     *     ViewIndex works (driver populates it per view).
     *   - If both halves identical → VS ViewIndex is always 0 (driver
     *     bug or pipeline not truly multiview).
     * This bypasses the normal stereo offset math entirely. */
    static int s_debug_vs_view = -1;
    if (s_debug_vs_view == -1) {
        const char *e = getenv("VKS3D_DEBUG_VS_VIEW");
        s_debug_vs_view = (e && e[0] == '1') ? 1 : 0;
    }
    if (c->have_view && m->view_var && m->ut && c->bt)
    {
        STEREO_LOG(
            "EMIT_BODY_VIEW have_view=%d view_var=%u uint_type=%u bool_type=%u — injecting ViewIndex offset",
            (int)c->have_view,
            m->view_var,
            m->ut,
            c->bt);
        if (s_debug_vs_view) {
            STEREO_LOG(
                "EMIT_BODY_DEBUG_VS_VIEW — replacing offset with ViewIndex*0.5 (diag mode)");
        }
        {
            STEREO_LOG_VERBOSE(
                "VIEW_LOAD "
                "type=%u "
                "ptr=%u "
                "ptrType=%u "
                "unsignedType=%u",
                m->ut,
                m->view_var,
                m->ptr_in_int,
                m->ut);
            uint32_t w[] = {
                op_(SpvOpLoad, 4),
                m->ut,
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
                /* SpvOpSelect condition ? true_val : false_val.
                 * When flip_eyes=1, stereo.c swaps cl/cr so that cl=-sep/2
                 * and cr=+sep/2.  The conv-term select (below) always uses
                 * +conv*w for isl=true and -conv*w for isl=false.  To keep
                 * sep and conv terms sign-aligned (both positive for one eye,
                 * both negative for the other), we must swap cl/cr here so
                 * that the eye with positive conv also gets positive sep.
                 *
                 * Result after subtraction:
                 *   L: nx = px + cr - conv*w  = px + sep/2 - conv*w
                 *   R: nx = px - cl + conv*w  = px - sep/2 + conv*w
                 *   Δ = ndc_L - ndc_R = sep/w - 2*conv
                 *     zero parallax at w = sep/(2*conv) */
                op_(SpvOpSelect, 6),
                m->ft,
                sel,
                isl,
                s_debug_vs_view ? c->cf0 : c->cr,
                s_debug_vs_view ? c->cr  : c->cl
            };
            sb_push_n(out, w, 6);
        }
        /* In debug mode (VKS3D_DEBUG_VS_VIEW=1), the OpSelect above already
         * produces sel = isl ? c->cf0 : c->cr, giving:
         *   ViewIndex=0 (isl=true)  → sel = 0.0   (no horizontal shift)
         *   ViewIndex=1 (isl=false) → sel = c->cr (~0.05 NDC, visible shift)
         * If the conv term is also applied (OFF_AXIS path below), the total
         * offset still differs between eyes, so left/right halves will be
         * visually different IF gl_ViewIndex is populated per-view.
         * If both halves are identical → gl_ViewIndex is always 0. */
    }
    else
    {
        STEREO_LOG(
            "EMIT_BODY_NOVIEW have_view=%d view_var=%u uint_type=%u bool_type=%u — using static offset only (FAIL: %s%s%s%s)",
            (int)c->have_view,
            m->view_var,
            m->ut,
            c->bt,
            !c->have_view ? " NO_VIEW" : "",
            !m->view_var ? " NO_VAR" : "",
            !m->ut ? " NO_UINT" : "",
            !c->bt ? " NO_BOOL" : "");
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
    STEREO_LOG_VERBOSE(
        "VS_PATCH "
        "mode=%d "
        "posVar=%u "
        "viewVar=%u "
        "block=%u "
        "member=%u "
        "haveView=%u "
        "convConst=%u "
        "leftConst=%u "
        "rightConst=%u",
        c->projection_mode,
        m->pos_var,
        m->view_var,
        m->pos_is_block,
        m->pos_member_idx,
        c->have_view,
        c->cc,
        c->cl,
        c->cr);
    STEREO_LOG_VERBOSE(
        "VS_PATCH_IDS "
        "ch=%u "
        "lp=%u "
        "lv=%u "
        "isl=%u "
        "sel=%u "
        "px=%u "
        "nx=%u "
        "np=%u "
        "mode=%d "
        "pos_var=%u "
        "pptr=%u "
        "view_var=%u "
        "leftConst=%u "
        "rightConst=%u "
        "convConst=%u",
        ch,
        lp,
        lv,
        isl,
        sel,
        px,
        nx,
        np,
        c->projection_mode,
        m->pos_var,
        pptr,
        m->view_var,
        c->cl,
        c->cr,
        c->cc);
    if (c->projection_mode == STEREO_PROJECTION_PARALLEL)
    {
        /* Clip-space constant offset.
         *
         * NDC offset = sel / w → depth-dependent:
         *   near objects (small w) → large screen shift
         *   far objects  (large w) → small screen shift
         *
         * This is mathematically equivalent to translating the
         * camera (eye offset in view space), which is the correct
         * way to produce stereo parallax.
         *
         * Convergence is NOT applied in this mode (use OFF_AXIS
         * for convergence support).
         */
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
        /* Depth source selection for conv term:
         * - PC games typically write a real perspective w to gl_Position.w
         *   (w = -z_view), so member 3 (w) is the correct depth proxy.
         * - Eden/Switch emulator shaders never write PositionW (the GPU
         *   implicitly uses w=1.0).  Eden's EmitPrologue initializes
         *   gl_Position = vec4(0,0,0,1) and only x/y/z are overwritten,
         *   leaving w as the constant 1.0.  Using w here would make the
         *   conv term constant, zeroing out parallax on both eyes.
         *   Fall back to gl_Position.z (member 2, NDC z in [0,1] after
         *   ConvertDepthMode) which actually varies with depth. */
        uint32_t depth_member = m->pos_w_is_const ? 2u : 3u;
        STEREO_LOG(
            "EMIT_BODY_DEPTH_SOURCE pos_w_is_const=%u using_member=%u (2=z NDC depth fallback for Eden; 3=w clip-space)",
            (unsigned)m->pos_w_is_const,
            (unsigned)depth_member);
        {
            uint32_t w[] = {
                op_(SpvOpCompositeExtract, 5),
                m->ft,
                pw,
                lp,
                depth_member
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
                op_(SpvOpFSub, 5),
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
                op_(SpvOpFSub, 5),
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
    STEREO_LOG_VERBOSE(
        "PROJ_WRITE pos_var=%u pptr=%u new_pos=%u x=%u view=%u",
        m->pos_var,
        pptr,
        np,
        nx,
        m->view_var);
    STEREO_LOG_VERBOSE(
        "VIEWSPACE_PATCH "
        "mode=%d "
        "patching_outPos=%u "
        "projection_found=%u "
        "memberMask=0x%X",
        c->projection_mode,
        1,
        m->proj_found,
        c->dbg ? c->dbg->proj_member_mask : 0);
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
    const uint32_t *in,
    size_t in_c,
    uint32_t **out,
    size_t *out_c,
    float lo,
    float ro,
    float conv,
    bool inj_vi,
    StereoDebugCtx *dbg)
{
    STEREO_LOG_VERBOSE("CALLED spirv_patch_stereo_vertex");
    if (!in || in_c < 5 || in[0] != SPIRV_MAGIC)
        return false;
    int projection_mode =
        cfg ? cfg->projection : STEREO_PROJECTION_PARALLEL;
    SpvMod m = {0};
    uint64_t spv_hash_summary = 0;
    bool     summary_counted = false;
    InterlockedIncrement((volatile long*)&g_stat_shaders_total);
    m.words = in;
    m.count = in_c;
    m.bound = m.words[3];
    m.value_capacity = m.bound + 64;
    m.value_from_matrix =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_matrix_type =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_matrix_ptr =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_proj_value =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_view_value =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_io_value =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.is_io_ptr =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.ac_base =
        calloc(m.value_capacity, sizeof(uint32_t));
    m.is_const_value =
        calloc(m.value_capacity, sizeof(uint8_t));
    m.vec4_w_id =
        calloc(m.value_capacity, sizeof(uint32_t));
    m.mtv_vec_w_const =
        calloc(m.value_capacity, sizeof(uint8_t));
    if (!m.value_from_matrix ||
        !m.is_matrix_type ||
        !m.is_matrix_ptr ||
        !m.is_proj_value ||
        !m.is_view_value ||
        !m.is_io_value ||
        !m.is_io_ptr ||
        !m.ac_base ||
        !m.is_const_value ||
        !m.vec4_w_id ||
        !m.mtv_vec_w_const)
    {
        free_spv_provenance(&m);
        return false;
    }
    /* Analyze shader structure before modification:
     * - matrix provenance
     * - gl_Position location
     * - ViewIndex availability
     * - entry point classification
     */
    spv_scan(&m);
    STEREO_LOG_VERBOSE(
        "VS_SCAN bound=%u it=%u ptr_in_int=%u view=%u",
        m.bound,
        m.it,
        m.ptr_in_int,
        m.view_var);
    /*
     * Optional shader blacklist.
     * Used for debugging shaders that should remain untouched.
     */
    uint64_t spv_hash = hash_spv(m.words, m.count);
    spv_hash_summary = spv_hash;

    //if (m.proj_found && m.proj_member_mask == 0x5)
    //{
    //    projection_mode = STEREO_PROJECTION_OFF_AXIS;
    //    STEREO_LOG_VERBOSE(
    //        "PROJ_FIXUP forcing off-axis projection hash=%016llx mask=0x%X",
    //        (unsigned long long)spv_hash,
    //        m.proj_member_mask);
    //}

    /* Reject trivial passthrough vertex shaders.
     *
     * Original heuristic: require has_matrix_ops (OpMatrixTimesVector
     * etc.) to prove the shader does projection.  This fails on
     * emulators (Yuzu/eden) that decompose mat4*vec4 into OpFMul/
     * OpFAdd/OpFma, producing no matrix-type SPIR-V ops at all.
     *
     * New heuristic: use pos_var (gl_Position output exists) as the
     * gate for ALL projection modes.  The OFF_AXIS patch only needs
     * gl_Position.x and .w — it does not need to locate the projection
     * matrix in a UBO struct.  True fullscreen-quad VS (hardcoded
     * positions, no UBO access) will still be filtered by the
     * SCREENSPACE_UI check (for non-PARALLEL modes) or will produce
     * harmless minor offsets.
     */
    if (m.exec_model == SpvExecVertex)
    {
        if (!m.pos_var)
        {
            InterlockedIncrement(
                (volatile long*)&g_stat_shaders_skip_no_mx);
            STEREO_LOG(
                "SHADER_SKIP hash=%016llx reason=NO_POS_VAR exec=%u "
                "proj_found=%u matrix_ops=%u dot=%u",
                (unsigned long long)spv_hash,
                (unsigned)m.exec_model,
                (unsigned)m.proj_found,
                (unsigned)m.has_matrix_ops,
                m.dot_count);
            free_spv_provenance(&m);
            return false;
        }
    }
    {
        static bool skip_list_init;
        static char skip_list[1024];
        if (!skip_list_init)
        {
            const char *env =
                stereo_getenv("VKS3D_SKIP_SHADER_PATCHES");
            if (env)
            {
                strncpy(skip_list, env, sizeof(skip_list) - 1);
                skip_list[sizeof(skip_list) - 1] = '\0';
            }
            skip_list_init = true;
        }
        if (skip_list[0])
        {
            char hashstr[17];
            snprintf(
                hashstr,
                sizeof(hashstr),
                "%016llx",
                (unsigned long long)spv_hash);
            if (strstr(skip_list, hashstr))
            {
                STEREO_LOG_VERBOSE(
                    "SKIP_SHADER_PATCH hash=%s",
                    hashstr);
                free_spv_provenance(&m);
                return false;
            }
        }
    }
    /*
     * Reject known monoscopic screen-space shaders.
     *
     * A fullscreen-quad / UI shader writes gl_Position from
     * compile-time constants only (e.g. vec4(-1,-1,0,1)).
     * It never reads vertex attributes, UBO, SSBO, or push
     * constants — the IO provenance array captures all of these.
     *
     * has_direct_position_write (set when the stored position
     * value has no MAT and no IO tag) is the sole criterion.
     * vertex_binding_count and is_quad are deliberately NOT
     * used because Yuzu/eden may pass vertex data through UBO
     * or SSBO instead of traditional vertex input bindings,
     * causing vertex_binding_count==0 even on real geometry VS.
     */
    if (cfg && cfg->mono_ui)
    {
        bool ui_candidate =
            m.has_direct_position_write &&
            !m.has_geom_pos_store &&
            !m.has_emit_vertex &&
            m.exec_model == SpvExecVertex;
        if (ui_candidate)
        {
            InterlockedIncrement((volatile long*)&g_stat_shaders_skip_ui);
            
            /* Dump skipped vertex shaders to numbered .spv files.
             * These files can be disassembled with spirv-dis to
             * understand why IO provenance failed. */
            {
                static uint32_t skip_dump_count = 0;
                if (skip_dump_count < 5) {
                    char fname[128];
                    snprintf(
                        fname, sizeof(fname),
                        "skip_shader_%02u_%016llx.spv",
                        (unsigned)skip_dump_count,
                        (unsigned long long)spv_hash);
                    FILE *fp = fopen(fname, "wb");
                    if (fp) {
                        fwrite(m.words, sizeof(uint32_t), m.count, fp);
                        fclose(fp);
                    }
                    skip_dump_count++;
                }
            }
            
            STEREO_LOG(
                "SHADER_SKIP hash=%016llx reason=SCREENSPACE_UI exec=%u "
                "proj_found=%u bindings=%u quad=%u src_id=%u",
                (unsigned long long)spv_hash,
                (unsigned)m.exec_model,
                (unsigned)m.proj_found,
                dbg ? (unsigned)dbg->vertex_binding_count : 0u,
                dbg ? (unsigned)dbg->is_quad : 0u,
                (unsigned)m.position_source_id);
            free_spv_provenance(&m);
            return false;
        }
    }
    if (dbg && !dbg->is_multiview)
    {
        InterlockedIncrement((volatile long*)&g_stat_shaders_skip_no_mv);
        STEREO_LOG(
            "SHADER_SKIP hash=%016llx reason=NON_MULTIVIEW_RP exec=%u "
            "proj_found=%u stage=%u",
            (unsigned long long)spv_hash,
            (unsigned)m.exec_model,
            (unsigned)m.proj_found,
            dbg ? (unsigned)dbg->stage : 0u);
        free_spv_provenance(&m);
        return false;
    }
    STEREO_LOG_VERBOSE(
        "PATCH_ANALYSIS hash=%016llx exec=%u patchable=%d pos=%u block=%d member=%u view=%u matrix=%d direct=%d dots=%u emit=%u mv=%d",
        (unsigned long long)spv_hash,
        m.exec_model,
        m.is_patchable,
        m.pos_var,
        m.pos_is_block,
        m.pos_member_idx,
        m.view_var,
        m.has_matrix_ops,
        m.has_direct_position_write,
        m.dot_count,
        m.emit_count,
        m.has_viewindex_builtin);
    STEREO_LOG_VERBOSE(
        "PROJ_FINAL hash=%016llx exec=%u found=%u set=%u binding=%u mask=0x%X access=%u loads=%u mtv=%u",
        (unsigned long long)spv_hash,
        m.exec_model,
        m.proj_found,
        m.proj_set,
        m.proj_binding,
        m.proj_member_mask,
        m.proj_access_count,
        m.proj_load_count,
        m.proj_mtv_count);
    STEREO_LOG_VERBOSE(
        "PROJ_DESCRIPTOR hash=%016llx struct=%u ptr=%u var=%u set=%u binding=%u",
        (unsigned long long)spv_hash,
        m.proj_struct_type,
        m.proj_ptr_type,
        m.proj_var,
        m.proj_set,
        m.proj_binding);
    if (m.proj_found)
    {
        InterlockedIncrement((volatile long*)&g_stat_proj_found);
        STEREO_LOG(
            "PROJ_FINAL_RESULT hash=%016llx FOUND=true source=OpTypeMatrix "
            "set=%u binding=%u struct_type=%u var=%u access_count=%u",
            (unsigned long long)spv_hash,
            m.proj_set,
            m.proj_binding,
            m.proj_struct_type,
            m.proj_var,
            m.proj_access_count);
    }
    else if (m.has_matrix_ops)
    {
        /* No OpTypeMatrix UBO found, but heuristic analysis detected
         * decomposed matrix multiplication (OpDot chain or GLSL.std.450
         * Fma chain).  This is the 3D scene vertex shader on Eden/Yuzu
         * where mat4*vec4 is lowered to scalar Fma/Dot operations. */
        m.proj_found = VK_TRUE;
        InterlockedIncrement((volatile long*)&g_stat_proj_found);
        const char *src =
            (m.fma_count >= 12) ? "FmaChain" :
            (m.dot_count >= 4)  ? "DotChain" : "Unknown";
        STEREO_LOG(
            "PROJ_FINAL_RESULT hash=%016llx FOUND=true source=%s "
            "dot=%u fma=%u mtv=%u (heuristic: decomposed mat4*vec4 "
            "via OpDot/GLSL.std.450 Fma, not OpTypeMatrix UBO)",
            (unsigned long long)spv_hash,
            src,
            m.dot_count,
            m.fma_count,
            m.proj_mtv_count);
    }
    else
    {
        InterlockedIncrement((volatile long*)&g_stat_proj_miss);
        STEREO_LOG(
            "PROJ_FINAL_RESULT hash=%016llx FOUND=false "
            "reason=NO_MATRIX_TYPE (no OpTypeMatrix, no Dot/Fma chain) "
            "struct=%u ptr=%u var=%u dot=%u fma=%u",
            (unsigned long long)spv_hash,
            m.proj_struct_type,
            m.proj_ptr_type,
            m.proj_var,
            m.dot_count,
            m.fma_count);
    }
    /* Dump shaders for offline spirv-dis/spirv-cross analysis.
     * Disabled by default to avoid disk I/O overhead during gameplay.
     * To enable: set VKS3D_DUMP_DIR env var to the desired output folder.
     * Files are named: dump_<FOUND>_<mtv>_<dot>_<hash>.spv */
    {
        const char *dump_dir = stereo_getenv("VKS3D_DUMP_DIR");
        if (dump_dir)
        {
            static uint32_t found_idx = 0;
            static uint32_t miss_idx = 0;
            static uint64_t dumped_found[16] = {0};
            static uint64_t dumped_miss[16] = {0};
            uint64_t *dumped_arr = m.proj_found ? dumped_found : dumped_miss;
            uint32_t *idx_ptr = m.proj_found ? &found_idx : &miss_idx;
            uint32_t cap = 16;
            bool already = false;
            for (uint32_t k = 0; k < cap && k < *idx_ptr; ++k)
            {
                if (dumped_arr[k] == spv_hash) { already = true; break; }
            }
            if (!already && *idx_ptr < cap)
            {
                char dpath[512];
                _snprintf(
                    dpath, sizeof(dpath) - 1,
                    "%s\\dump_%s_mtv%u_dot%u_%016llx.spv",
                    dump_dir,
                    m.proj_found ? "FOUND" : "MISS",
                    m.proj_mtv_count,
                    m.dot_count,
                    (unsigned long long)spv_hash);
                FILE *fp = fopen(dpath, "wb");
                if (fp)
                {
                    fwrite(m.words, sizeof(uint32_t), m.count, fp);
                    fclose(fp);
                    STEREO_LOG(
                        "SHADER_DUMPED hash=%016llx path=%s "
                        "proj_found=%u mtv=%u dot=%u matrix_ops=%u",
                        (unsigned long long)spv_hash,
                        dpath,
                        m.proj_found ? 1u : 0u,
                        m.proj_mtv_count,
                        m.dot_count,
                        m.has_matrix_ops ? 1u : 0u);
                }
                dumped_arr[(*idx_ptr)++] = spv_hash;
            }
        }
    }
    if (dbg)
    {
        dbg->has_matrix_ops = m.has_matrix_ops;
        dbg->direct_position_write = m.has_direct_position_write;
        dbg->has_proj_ubo = false;
        if (m.proj_found)
        {
            dbg->has_proj_ubo = true;
            dbg->proj_set = m.proj_set;
            dbg->proj_binding = m.proj_binding;
            dbg->proj_member_mask = m.proj_member_mask;
            dbg->proj_var = m.proj_var;
        }
        STEREO_LOG_VERBOSE(
            "PROJ_DETECT hash=%016llx found=%u set=%u binding=%u mask=0x%X var=%u",
            (unsigned long long)spv_hash,
            m.proj_found,
            dbg->proj_set,
            dbg->proj_binding,
            dbg->proj_member_mask,
            dbg->proj_var);
        STEREO_LOG_VERBOSE(
            "PROJ_TRACE access_count=%u load_count=%u mtv_count=%u mask=0x%X",
            m.proj_access_count,
            m.proj_load_count,
            m.proj_mtv_count,
            m.proj_member_mask);
    }
    if (m.exec_model == SpvExecVertex)
    {
        if (!m.pos_var)
        {
            InterlockedIncrement((volatile long*)&g_stat_shaders_skip_no_pos);
            STEREO_LOG(
                "SHADER_SKIP hash=%016llx reason=NO_POSITION exec=%u "
                "proj_found=%u patchable=%u",
                (unsigned long long)spv_hash,
                (unsigned)m.exec_model,
                (unsigned)m.proj_found,
                (unsigned)m.is_patchable);
            free_spv_provenance(&m);
            return false;
        }
        /* OFF_AXIS depth-dependent gate.
         *
         * OFF_AXIS patch computes convergence as convmag = pw * cc,
         * where pw = gl_Position.w.  This requires pw to be the proper
         * post-projection depth value (-view_space_z).  If the shader
         * does not multiply by a projection matrix (no OpMatrixTimesVector,
         * no dot_count>=4 decomposed mat4*vec4, no proj_found), pw is
         * likely a constant (1.0), making the convergence offset constant
         * instead of depth-dependent — this produces a uniform whole-screen
         * shift rather than proper parallax.
         *
         * Skip such shaders for OFF_AXIS; let them render monoscopically.
         * Shaders with proj_found=1 OR has_matrix_ops=true OR dot_count>=4
         * are considered to perform proper projection.
         *
         * PARALLEL mode is left unchanged (it uses only sel, no pw). */
        if (projection_mode == STEREO_PROJECTION_OFF_AXIS &&
            !m.has_matrix_ops &&
            !m.proj_found &&
            m.dot_count < 4 &&
            m.fma_count < 12)
        {
            InterlockedIncrement((volatile long*)&g_stat_shaders_skip_ui);
            STEREO_LOG(
                "SHADER_SKIP hash=%016llx reason=NO_PROJECTION_MUL exec=%u "
                "proj_found=%u matrix_ops=%u dot=%u fma=%u mtv=%u (OFF_AXIS needs proper depth in gl_Position.w)",
                (unsigned long long)spv_hash,
                (unsigned)m.exec_model,
                (unsigned)m.proj_found,
                (unsigned)m.has_matrix_ops,
                m.dot_count,
                m.fma_count,
                m.proj_mtv_count);
            free_spv_provenance(&m);
            return false;
        }
        /* SCREENSPACE_BLOCK guard.
         *
         * Historically this rejected shaders where gl_Position is written
         * through a block member (pos_is_block=true) AND no matrix ops
         * were found.  The idea was to skip UI / reconstruction shaders
         * that read a pre-projected position from a G-buffer and write
         * it through an Output block without any MVP multiply.
         *
         * This guard was intended for projection modes that rewrite the
         * projection matrix inside a UBO (which needs to locate the
         * matrix type).  OFF_AXIS and PARALLEL only rewrite the
         * gl_Position.x component after the fact — they do not need to
         * understand the UBO layout or matrix types at all.  Therefore
         * skip this gate entirely when matrix-struct awareness is not
         * required.
         *
         * Yuzu/eden specifically have pos_is_block=false for all VS
         * because of the flat float[] cbuf layout, so this check was
         * never a blocker there — but we still remove the unnecessary
         * dependency on has_matrix_ops for correctness. */
        if (m.pos_is_block && !m.has_matrix_ops &&
            (projection_mode == STEREO_PROJECTION_OFF_AXIS ||
             projection_mode == STEREO_PROJECTION_PARALLEL))
        {
            STEREO_LOG_VERBOSE(
                "SCREENSPACE_BLOCK_SKIPPED hash=%016llx "
                "pos_block=%u matrix_ops=%u mode=%d (mode does not need UBO matrix lookup)",
                (unsigned long long)spv_hash,
                (unsigned)m.pos_is_block,
                (unsigned)m.has_matrix_ops,
                (int)projection_mode);
        }
        else if (m.pos_is_block && !m.has_matrix_ops)
        {
            InterlockedIncrement((volatile long*)&g_stat_shaders_skip_ui);
            STEREO_LOG(
                "SHADER_SKIP hash=%016llx reason=SCREENSPACE_BLOCK exec=%u "
                "proj_found=%u",
                (unsigned long long)spv_hash,
                (unsigned)m.exec_model,
                (unsigned)m.proj_found);
            free_spv_provenance(&m);
            return false;
        }
    }
    if (!m.is_patchable)
    {
        InterlockedIncrement((volatile long*)&g_stat_shaders_skip_unpch);
        STEREO_LOG(
            "SHADER_SKIP hash=%016llx reason=UNPATCHABLE exec=%u "
            "proj_found=%u pos=%u matrix=%u",
            (unsigned long long)spv_hash,
            (unsigned)m.exec_model,
            (unsigned)m.proj_found,
            m.pos_var,
            (unsigned)m.has_matrix_ops);
        free_spv_provenance(&m);
        return false;
    }
    /* Allocate new SPIR-V IDs and prepare injected objects:
     * - output position pointer
     * - ViewIndex input
     * - stereo constants
     * - temporary types
     *
     * Future projection-matrix patching should extend this stage.
     */
    bool is_gs =
        (m.exec_model == SpvExecGeometry);
    uint32_t nid = m.bound;
    uint32_t id_ptr_v4 = nid++;
    uint32_t id_ptr_int = nid++;
    uint32_t id_new_ut = 0;
    if (!m.ut && inj_vi && !m.view_var)
    {
        id_new_ut = nid++;
        m.ut = id_new_ut;
    }
    STEREO_LOG(
        "PATCH_PREP hash=%016llx exec=%u proj_found=%u proj_set=%u proj_binding=%u view_var=%u ut=%u bt=%u",
        (unsigned long long)spv_hash,
        (unsigned)m.exec_model,
        (unsigned)m.proj_found,
        m.proj_found ? m.proj_set : UINT32_MAX,
        m.proj_found ? m.proj_binding : UINT32_MAX,
        m.view_var,
        m.ut,
        m.bt);

    bool will_inj_vi =
        inj_vi &&
        !m.view_var &&
        m.ut;
    uint32_t id_inj_view =
        will_inj_vi ? nid++ : 0;
    bool have_view =
        m.view_var ||
        will_inj_vi;
    STEREO_LOG(
        "PATCH_VIEWINFO will_inj_vi=%d have_view=%d id_inj_view=%u new_ut=%u",
        (int)will_inj_vi,
        (int)have_view,
        id_inj_view,
        id_new_ut);
    uint32_t id_new_bt = 0;
    if (!m.bt &&
        !m.bt_type &&
        have_view &&
        m.ut)
    {
        id_new_bt = nid++;
    }
    uint32_t id_cz = nid++;
    uint32_t id_cf0 = nid++;
    uint32_t id_cl = nid++;
    uint32_t id_cr = nid++;
    uint32_t id_cc = nid++;
    STEREO_LOG_VERBOSE(
        "VS_NEW_IDS "
        "bound=%u "
        "next=%u "
        "ptr_v4=%u "
        "ptr_int=%u "
        "new_ut=%u "
        "inj_view=%u "
        "new_bool=%u "
        "cz=%u "
        "cf0=%u "
        "cl=%u "
        "cr=%u "
        "cc=%u",
        m.bound,
        nid,
        id_ptr_v4,
        id_ptr_int,
        id_new_ut,
        id_inj_view,
        id_new_bt,
        id_cz,
        id_cf0,
        id_cl,
        id_cr,
        id_cc);
    uint32_t uv4 =
        m.ptr_out_v4 ?
        m.ptr_out_v4 :
        id_ptr_v4;
    uint32_t uint_ =
        m.ptr_in_int ?
        m.ptr_in_int :
        id_ptr_int;
    uint32_t bt =
        m.bt ?
        m.bt :
        (m.bt_type ? m.bt_type : id_new_bt);
    /* Additional SPIR-V declarations inserted before the entry function:
     * - new types
     * - constants
     * - ViewIndex variable
     */
    SpvBuf ann;
    SpvBuf te;
    if (!sb_init(&ann, 16) || !sb_init(&te, 96))
    {
        free_spv_provenance(&m);
        return false;
    }
    if (id_new_ut)
    {
        uint32_t w[] =
        {
            op_(SpvOpTypeInt, 4),
            id_new_ut,
            32,
            0
        };
        sb_push_n(&te, w, 4);
    }
    if (!m.ptr_out_v4)
    {
        uint32_t w[] =
        {
            op_(SpvOpTypePointer, 4),
            id_ptr_v4,
            SpvStorageOutput,
            m.v4t
        };
        sb_push_n(&te, w, 4);
    }
    if (m.ut && !m.ptr_in_int)
    {
        uint32_t w[] =
        {
            op_(SpvOpTypePointer, 4),
            id_ptr_int,
            SpvStorageInput,
            m.ut
        };
        sb_push_n(&te, w, 4);
        m.ptr_in_int = id_ptr_int;
        uint_ = id_ptr_int;
    }
    if (id_new_bt)
    {
        uint32_t w[] =
        {
            op_(SpvOpTypeBool, 2),
            id_new_bt
        };
        sb_push_n(&te, w, 2);
    }
    if (m.ut)
    {
        uint32_t w[] =
        {
            op_(SpvOpConstant, 4),
            m.ut,
            id_cz,
            0
        };
        sb_push_n(&te, w, 4);
    }
    {
        uint32_t w[4] =
        {
            op_(SpvOpConstant, 4),
            m.ft,
            id_cf0,
            0
        };
        float z = 0.0f;
        memcpy(&w[3], &z, sizeof(z));
        sb_push_n(&te, w, 4);
    }
    {
        uint32_t w[4] =
        {
            op_(SpvOpConstant, 4),
            m.ft,
            id_cl,
            0
        };
        memcpy(&w[3], &lo, sizeof(lo));
        sb_push_n(&te, w, 4);
    }
    {
        uint32_t w[4] =
        {
            op_(SpvOpConstant, 4),
            m.ft,
            id_cr,
            0
        };
        memcpy(&w[3], &ro, sizeof(ro));
        sb_push_n(&te, w, 4);
    }
    {
        uint32_t w[4] =
        {
            op_(SpvOpConstant, 4),
            m.ft,
            id_cc,
            0
        };
        memcpy(&w[3], &conv, sizeof(conv));
        sb_push_n(&te, w, 4);
    }
    if (will_inj_vi)
    {
        uint32_t d[] =
        {
            op_(SpvOpDecorate, 4),
            id_inj_view,
            SpvDecorationBuiltIn,
            SpvBuiltInViewIndex
        };
        sb_push_n(&ann, d, 4);
        STEREO_LOG(
            "INJ_VIEW_DECORATE id=%u builtin=ViewIndex",
            id_inj_view);
        uint32_t v[] =
        {
            op_(SpvOpVariable, 4),
            uint_,
            id_inj_view,
            SpvStorageInput
        };
        sb_push_n(&te, v, 4);
        m.view_var = id_inj_view;
        STEREO_LOG(
            "INJ_VIEW_VAR id=%u ptr_type=%u storage=Input",
            id_inj_view,
            uint_);
    }
    BodyCtx bc =
    {
        .m                   = &m,
        .have_view           = have_view,
        .has_projection_path = false,
        .has_view_path       = false,
        .uv4                 = uv4,
        .uint_               = uint_,
        .bt                  = bt,
        .cz                  = id_cz,
        .cf0                 = id_cf0,
        .cl                  = id_cl,
        .cr                  = id_cr,
        .cc                  = id_cc,
        .projection_mode     = projection_mode,
        .lo_dbg              = lo,
        .ro_dbg              = ro,
        .conv_dbg            = conv,
        .dbg                 = dbg
    };
    /* Vertex/TessEval shaders:
     * inject after final position calculation.
     *
     * Geometry shaders:
     * inject before EmitVertex.
     */
    size_t ins_ann = 0;
    size_t ins_t = 0;
    size_t ins_te = 0;
    size_t ins_b = 0;
    bool in_entry_function = false;
    for (size_t i = 5; i < in_c;)
    {
        uint32_t opx = in[i] & 0xffff;
        uint32_t wcx = in[i] >> 16;
        if (!wcx || i + wcx > in_c)
            break;
        if (opx == SpvOpVariable && wcx >= 4)
        {
            STEREO_LOG_VERBOSE(
                "VS_VARIABLE_IN result=%u type=%u storage=%u",
                in[i + 2],
                in[i + 1],
                in[i + 3]);
        }
        if (!ins_ann &&
            (opx == SpvOpTypeVoid ||
             opx == SpvOpTypeBool ||
             opx == SpvOpTypeInt ||
             opx == SpvOpTypeFloat ||
             opx == SpvOpTypeVector ||
             opx == SpvOpTypeMatrix ||
             opx == SpvOpTypeImage ||
             opx == SpvOpTypeSampler ||
             opx == SpvOpTypeSampledImage ||
             opx == SpvOpTypeArray ||
             opx == SpvOpTypeRuntimeArray ||
             opx == SpvOpTypeStruct ||
             opx == SpvOpTypeOpaque ||
             opx == SpvOpTypePointer ||
             opx == SpvOpTypeFunction ||
             opx == SpvOpTypeForwardPointer ||
             opx == SpvOpConstantTrue ||
             opx == SpvOpConstantFalse ||
             opx == SpvOpConstant ||
             opx == SpvOpConstantComposite ||
             opx == SpvOpVariable))
        {
            ins_ann = i;
        }
        if (opx == SpvOpFunction)
        {
            if (!ins_te)
                ins_te = i;
            in_entry_function =
                (wcx >= 4 &&
                 in[i + 2] ==
                 (m.position_function ?
                  m.position_function :
                  m.entry_function));
            if (in_entry_function)
                ins_t = i;
        }
        if (in_entry_function &&
            opx == SpvOpReturn)
        {
            ins_b = i;
        }
        if (in_entry_function &&
            opx == SpvOpFunctionEnd)
        {
            break;
        }
        i += wcx;
    }
    STEREO_LOG(
        "PATCH_INJECT_POINTS hash=%016llx ins_ann=%zu ins_te=%zu ins_t=%zu ins_b=%zu "
        "fn_word=%zu entry_fn=%u pos_fn=%u",
        (unsigned long long)spv_hash,
        ins_ann,
        ins_te,
        ins_t,
        ins_b,
        m.fn_word,
        m.entry_function,
        m.position_function);

    if (!ins_t)
    {
        sb_free(&te);
        free_spv_provenance(&m);
        return false;
    }
    if (!is_gs && !ins_b)
    {
        sb_free(&te);
        free_spv_provenance(&m);
        return false;
    }
    bool need_mv_cap =
        (id_inj_view || m.view_var) &&
        !m.has_mv_cap;
    bool mv_done = false;
    bool ann_done = false;
    bool te_done = false;
    bool body_done = false;
    /* Rebuild the SPIR-V module:
     * - add MultiView capability if required
     * - insert declarations
     * - extend entry point interface
     * - inject stereo body
     */
    SpvBuf ob;
    if (!sb_init(&ob, in_c + te.n + 64))
    {
        sb_free(&te);
        free_spv_provenance(&m);
        return false;
    }
    STEREO_LOG_VERBOSE(
        "SPV_HEADER version=0x%08X generator=0x%08X bound=%u",
        in[1],
        in[2],
        in[3]);
    uint32_t spv_version = in[1];
    bool need_mv_ext =
        need_mv_cap &&
        ((spv_version >> 16) == 1) &&
        (((spv_version >> 8) & 0xff) == 0);
    STEREO_LOG(
        "SPV_MULTIVIEW need_cap=%d need_ext=%d has_mv_cap=%d view_var=%u inj_view=%u version=0x%08X bound=%u",
        (int)need_mv_cap,
        (int)need_mv_ext,
        (int)m.has_mv_cap,
        m.view_var,
        id_inj_view,
        in[1],
        in[3]);
    /* Header only. We'll inject after the last OpCapability. */
    sb_push_n(&ob, in, 5);
    bool ext_done = false;
    for (size_t i = 5; i < in_c;)
    {
        if (!mv_done && need_mv_cap)
        {
            uint32_t c[] =
            {
                op_(SpvOpCapability, 2),
                SpvCapabilityMultiView
            };
            sb_push_n(&ob, c, 2);
            mv_done = true;
            STEREO_LOG(
                "INJ_MULTIVIEW_CAP emitted MultiView capability (id_inj_view=%u has_mv_cap=%d)",
                id_inj_view,
                (int)m.has_mv_cap);
        }
        if (!ann_done && i == ins_ann)
        {
            sb_push_n(&ob, ann.w, ann.n);
            ann_done = true;
        }
        if (!te_done && ins_te && i == ins_te)
        {
            sb_push_n(&ob, te.w, te.n);
            te_done = true;
        }
        uint32_t opx = in[i] & 0xffff;
        uint32_t wcx = in[i] >> 16;
        if (!wcx || i + wcx > in_c)
            break;
        /* After the final OpCapability, emit OpExtension if required. */
        if (!ext_done &&
            need_mv_ext &&
            opx != SpvOpCapability)
        {
            uint32_t e[] =
            {
                op_(SpvOpExtension, 6),
                0x5F565053, /* SPV_ */
                0x5F52484B, /* KHR_ */
                0x746C756D, /* mult */
                0x65697669, /* ivie */
                0x00000077  /* w */
            };
            sb_push_n(&ob, e, 6);
            ext_done = true;
            STEREO_LOG(
                "INJ_MULTIVIEW_EXT emitted SPV_KHR_multiview extension (main patcher)");
        }
        if (id_inj_view &&
            opx == SpvOpEntryPoint &&
            wcx >= 4 &&
            (in[i + 1] == SpvExecVertex ||
             in[i + 1] == SpvExecGeometry ||
             in[i + 1] == SpvExecTessEval))
        {
            bool target_entry =
                (wcx >= 3 &&
                 in[i + 2] == m.entry_function);
            if (target_entry)
            {
                sb_push(
                    &ob,
                    ((wcx + 1) << 16) |
                    SpvOpEntryPoint);
                sb_push_n(
                    &ob,
                    &in[i + 1],
                    wcx - 1);
                sb_push(
                    &ob,
                    id_inj_view);
                STEREO_LOG(
                    "INJ_VIEW_IFACE entry=%u interface=%zu appended_view=%u",
                    in[i + 1],
                    (size_t)(wcx - 1),
                    id_inj_view);
            }
            else
            {
                sb_push_n(
                    &ob,
                    &in[i],
                    wcx);
            }
            i += wcx;
            continue;
        }
        STEREO_LOG_VERBOSE(
            "VS_PATCH_SUMMARY pipeline=%u projFound=%u viewBuiltin=%u projMode=%u "
            "projVar=%u members=0x%X matrixOps=%u directPos=%u",
            dbg ? dbg->pipeline_index : 0,
            m.proj_found,
            m.has_viewindex_builtin,
            projection_mode,
            dbg ? dbg->proj_var : 0,
            dbg ? dbg->proj_member_mask : 0,
            dbg ? dbg->has_matrix_ops : 0,
            dbg ? dbg->direct_position_write : 0);
        if (is_gs &&
            opx == SpvOpEmitVertex)
        {
            emit_body(
                &ob,
                &bc,
                &nid);
        }
        if (!is_gs &&
            !body_done &&
            i == ins_b)
        {
            emit_body(
                &ob,
                &bc,
                &nid);
            body_done = true;
        }
        sb_push_n(
            &ob,
            &in[i],
            wcx);
        i += wcx;
    }
    /* Module contained only capabilities before declarations. */
    if (!ext_done && need_mv_ext)
    {
        uint32_t e[] =
        {
            op_(SpvOpExtension, 6),
            0x5F565053,
            0x5F52484B,
            0x746C756D,
            0x65697669,
            0x00000077
        };
        sb_push_n(&ob, e, 6);
        STEREO_LOG(
            "INJ_MULTIVIEW_EXT emitted SPV_KHR_multiview extension (tail fallback)");
    }
    if (!ann_done)
        sb_push_n(&ob, ann.w, ann.n);
    if (!te_done)
        sb_push_n(&ob, te.w, te.n);
    /* Verify OpVariable declarations after rewriting */
    for (size_t j = 5; j < ob.n;)
    {
        uint32_t opj = ob.w[j] & 0xffff;
        uint32_t wcj = ob.w[j] >> 16;
        if (opj == SpvOpVariable &&
            wcj >= 4 &&
            ob.w[j + 2] == id_inj_view)
        {
            STEREO_LOG_VERBOSE(
                "VS_VIEW_VAR_DEF "
                "result=%u "
                "ptrType=%u "
                "storage=%u",
                ob.w[j + 2],
                ob.w[j + 1],
                ob.w[j + 3]);
        }
        if ((opj == SpvOpTypePointer ||
             opj == SpvOpTypeInt ||
             opj == SpvOpTypeVector) &&
            wcj >= 2)
        {
            STEREO_LOG_VERBOSE(
                "VS_TYPE_OUT opcode=%u (%s) id=%u",
                opj,
                spv_op_name(opj),
                ob.w[j + 1]);
        }
        if (!wcj || j + wcj > ob.n)
            break;
        if (opj == SpvOpVariable && wcj >= 4)
        {
            STEREO_LOG_VERBOSE(
                "VS_VARIABLE_OUT result=%u type=%u storage=%u",
                ob.w[j + 2],
                ob.w[j + 1],
                ob.w[j + 3]);
        }
        if (opj == SpvOpTypePointer && wcj >= 4)
        {
            STEREO_LOG_VERBOSE(
                "VS_TYPE_POINTER_OUT id=%u storage=%u pointee=%u",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3]);
        }
        if (opj == SpvOpTypeInt && wcj >= 4)
        {
            STEREO_LOG_VERBOSE(
                "VS_TYPE_INT id=%u width=%u signed=%u",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3]);
        }
        if (ob.w[j + 1] == ob.w[j + 1]) /* keep compiler happy */
        {
            if (ob.w[j + 1] == 16 ||
                ob.w[j + 1] == id_ptr_int)
            {
                STEREO_LOG_VERBOSE(
                    "VS_VIEW_POINTER ptr=%u storage=%u pointee=%u",
                    ob.w[j + 1],
                    ob.w[j + 2],
                    ob.w[j + 3]);
            }
        }
        if (opj == SpvOpVariable &&
            wcj >= 4 &&
            ob.w[j + 2] == id_inj_view)
        {
            STEREO_LOG_VERBOSE(
                "VS_VIEW_VAR type=%u storage=%u",
                ob.w[j + 1],
                ob.w[j + 3]);
        }
        j += wcj;
    }
    sb_free(&ann);
    sb_free(&te);
    ob.w[3] = nid;
    for (size_t j = 5; j < ob.n;)
    {
        uint32_t op = ob.w[j] & 0xffff;
        uint32_t wc = ob.w[j] >> 16;
        if (op == SpvOpLoad && wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "OUT_LOAD "
                "type=%u "
                "result=%u "
                "ptr=%u",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3]);
            if (ob.w[j + 3] == m.view_var)
            {
                STEREO_LOG_VERBOSE(
                    "VIEW_LOAD_FINAL type=%u ptr=%u expectedPtr=%u expectedType=%u",
                    ob.w[j + 1],
                    ob.w[j + 3],
                    m.view_var,
                    m.it);
            }
        }
        if (!wc || j + wc > ob.n)
            break;
        j += wc;
    }
    /* Update SPIR-V bound field (word 3) to reflect new IDs.
     * Without this, IDs like ViewIndex variable (id_inj_view) exceed
     * the original bound and the driver may ignore them. */
    if (nid > ob.w[3])
    {
        STEREO_LOG(
            "FIX_BOUND old=%u new=%u — updating SPIR-V bound to cover injected IDs",
            ob.w[3], nid);
        ob.w[3] = nid;
    }
    *out = ob.w;
    *out_c = ob.n;
    /*
     * Finalize patched SPIR-V module.
     * Provenance tables are no longer needed after reconstruction.
     */
    InterlockedIncrement((volatile long*)&g_stat_shaders_patch_ok);
    STEREO_LOG(
        "SHADER_PATCHED hash=%016llx exec=%u proj_found=%u "
        "set=%u binding=%u mask=0x%X mode=%u words_in=%zu words_out=%zu "
        "matrix_ops=%u dot=%u fma=%u mtv=%u",
        (unsigned long long)spv_hash_summary,
        (unsigned)m.exec_model,
        (unsigned)m.proj_found,
        m.proj_found ? m.proj_set : UINT32_MAX,
        m.proj_found ? m.proj_binding : UINT32_MAX,
        m.proj_found ? m.proj_member_mask : 0u,
        (unsigned)projection_mode,
        in_c,
        (size_t)ob.n,
        (unsigned)m.has_matrix_ops,
        m.dot_count,
        m.fma_count,
        m.proj_mtv_count);
    /* Dump patched SPIR-V for offline spirv-dis/spirv-cross analysis.
     *
     * Always dump first N patched shaders to the host EXE directory so
     * we can verify OFF_AXIS offset injection without setting env vars.
     * After N files, fall back to VKS3D_DUMP_DIR-only behaviour to avoid
     * disk I/O spam.  N is intentionally small (<=8) because every
     * unique shader hash will typically produce a ~20-200 KB SPIR-V file.
     */
    extern char g_exe_dir[512];
    {
        static volatile LONG s_patched_dumped = 0;
        const int PATCHED_DUMP_LIMIT = 8;
        bool force_dump = false;
        const char *dump_dir = stereo_getenv("VKS3D_DUMP_DIR");
        LONG slot = -1;
        if (!dump_dir) {
            slot = InterlockedExchangeAdd(&s_patched_dumped, 1);
            if (slot < PATCHED_DUMP_LIMIT)
                force_dump = true;
        }
        if (dump_dir || force_dump) {
            char dpath[512];
            const char *use_dir = dump_dir ? dump_dir : g_exe_dir;
            _snprintf(
                dpath, sizeof(dpath) - 1,
                "%s\\PATCHED_%016llx_slot%ld.spv",
                use_dir,
                (unsigned long long)spv_hash_summary,
                (long)slot);
            FILE *fp = fopen(dpath, "wb");
            if (fp) {
                fwrite(ob.w, sizeof(uint32_t), ob.n, fp);
                fclose(fp);
                STEREO_LOG(
                    "PATCHED_DUMPED hash=%016llx path=%s words=%zu slot=%ld/%d force=%d",
                    (unsigned long long)spv_hash_summary,
                    dpath,
                    (size_t)ob.n,
                    (long)slot,
                    PATCHED_DUMP_LIMIT,
                    (int)force_dump);
            } else {
                STEREO_LOG(
                    "PATCHED_DUMP_FAIL hash=%016llx dir=%s err=%d",
                    (unsigned long long)spv_hash_summary,
                    use_dir,
                    (int)errno);
            }
        }
    }
    free_spv_provenance(&m);
    return true;
}

void spirv_patched_free(uint32_t *w) { free(w); }

/* ══════════════════════════════════════════════════════════════════════════
 * Fragment shader analysis state
 *
 * Tracks descriptor ownership, sampled-image propagation and function
 * parameter forwarding so the patcher can determine which image accesses
 * should become array-layered. Future projection-matrix analysis will
 * extend this structure rather than introducing another scanner.
 * ══════════════════════════════════════════════════════════════════════════
 */

#define FS_MAX_IMG        256
#define FS_MAX_SI         256
#define FS_MAX_LOADS     2048
#define FS_MAX_PARAMS    1024
#define FS_MAX_CALLS     1024
#define FS_MAX_FUNCTIONS 256
#define FS_MAX_VARS       512

typedef struct
{
    uint32_t id;           /* OpLoad result id */
    uint32_t source_id;    /* Original source SSA id */
    uint32_t owner_var;    /* Descriptor variable owning this resource */
    uint32_t binding;      /* Cached binding after fixup */
    /* ---- Projection provenance ---- */
    bool     from_projection;
    bool     from_view;
} FsLoadInfo;

typedef struct
{
    uint32_t id;
    uint32_t type;
    uint32_t function_id;
    uint32_t index;
} FsParameterInfo;

typedef struct
{
    uint32_t id;
    uint32_t first_param;
    uint32_t type_id;
} FsFunctionInfo;

typedef struct
{
    uint32_t function_id;
    uint32_t parameter_index; /* temporary parameter slot */
    uint32_t parameter_id;    /* resolved parameter id */
    uint32_t argument_var;    /* SSA id passed to the call */
} FsCallInfo;

typedef struct
{
    uint32_t id;
    uint32_t type;
    uint32_t storage;
    uint32_t set;
    uint32_t binding;
    uint32_t location;
    bool     is_projection_ubo;
} FsVariableInfo;

typedef struct
{
    uint32_t target;
    uint32_t binding;
    uint32_t set;
    uint32_t location;
} FsDecorationInfo;

typedef struct
{
    uint32_t id;
    uint32_t sampled_type_id;
    uint32_t sampled_type;
    uint32_t dim;
    uint32_t depth;
    uint32_t arrayed;
    uint32_t ms;
    uint32_t sampled;
    uint32_t format;
    bool     patchable;
    uint32_t pointer_type;
    uint32_t owner_var;
    uint32_t binding;
    uint32_t set;
    bool     stereo;
    uint32_t replacement_type; /* existing array image type if reused */
    uint32_t replacement_pointer_type;
    uint32_t replacement_sampled_type;
} FsImageInfo;

typedef struct
{
    //Image type declarations
    FsImageInfo images[FS_MAX_IMG];
    uint32_t    n_img;
    //Sampled-image type declarations
    uint32_t si_ids[FS_MAX_SI];
    uint32_t n_si;
    //Resource ownership tables
    FsLoadInfo loads[FS_MAX_LOADS];
    uint32_t   n_load;
    FsParameterInfo params[FS_MAX_PARAMS];
    uint32_t        n_param;
    FsCallInfo calls[FS_MAX_CALLS];
    uint32_t   n_call;
    //Descriptor variables
    FsVariableInfo vars[FS_MAX_VARS];
    uint32_t       n_var;
    //Decorations
    FsDecorationInfo decorations[FS_MAX_VARS];
    uint32_t         n_dec;
    //Cached SPIR-V types
    uint32_t float_id;
    uint32_t int_id;
    uint32_t uint_id;
    uint32_t v2int_id;
    uint32_t v2uint_id;
    uint32_t v3int_id;
    uint32_t v3uint_id;
    uint32_t v3float_id;
    uint32_t ptr_int_in_id;
    uint32_t vi_var_id;
    bool     has_mv_cap;
    //Entry point information
    size_t ep_word;
    size_t fn_word;
    //Function table
    FsFunctionInfo functions[FS_MAX_FUNCTIONS];
    uint32_t       n_function;
    //Current function during prescan
    bool     in_function;
    uint32_t current_function_id;
    uint32_t current_param_index;
} FsScan;

static bool
fs_id_in(
    const uint32_t *arr,
    uint32_t n,
    uint32_t id)
{
    for (uint32_t i = 0; i < n; i++)
    {
        if (arr[i] == id)
            return true;
    }

    return false;
}

static int
fs_var_index(
    const FsScan *s,
    uint32_t id)
{
    if (!s)
        return -1;

    for (uint32_t i = 0; i < s->n_var; ++i)
    {
        if (s->vars[i].id == id)
            return (int)i;
    }

    return -1;
}

static void
fs_dump_descriptor_chain(
    const FsScan *s,
    const uint32_t *spv,
    size_t word_count,
    uint32_t descriptor_var)
{
    int vi = fs_var_index(s, descriptor_var);
    if (vi < 0)
        return;
    uint32_t type_id = s->vars[vi].type;
    STEREO_LOG_VERBOSE(
        "FS_RESOURCE descriptor=%u varType=%u",
        descriptor_var,
        type_id);
    for (size_t i = 5; i < word_count; )
    {
        uint32_t wc = spv[i] >> 16;
        uint32_t op = spv[i] & 0xffff;
        if (!wc || i + wc > word_count)
            break;
        if ((op == SpvOpTypePointer ||
             op == SpvOpTypeSampledImage ||
             op == SpvOpTypeImage) &&
            spv[i + 1] == type_id)
        {
            STEREO_LOG_VERBOSE(
                "FS_RESOURCE_TYPE id=%u op=%s",
                type_id,
                spv_op_name(op));
            if (op == SpvOpTypePointer && wc >= 4)
            {
                STEREO_LOG_VERBOSE(
                    "FS_POINTER elementType=%u storage=%u",
                    spv[i + 3],
                    spv[i + 2]);
                type_id = spv[i + 3];
                i = 5;
                continue;
            }
            if (op == SpvOpTypeSampledImage && wc >= 3)
            {
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLED_IMAGE imageType=%u",
                    spv[i + 2]);
                type_id = spv[i + 2];
                i = 5;
                continue;
            }
            if (op == SpvOpTypeImage && wc >= 9)
            {
                STEREO_LOG_VERBOSE(
                    "FS_IMAGE_TYPE sampledType=%u dim=%u depth=%u arrayed=%u ms=%u sampled=%u format=%u",
                    spv[i + 2],
                    spv[i + 3],
                    spv[i + 4],
                    spv[i + 5],
                    spv[i + 6],
                    spv[i + 7],
                    spv[i + 8]);
                break;
            }
        }
        i += wc;
    }
}

static int
fs_dec_index(
    const FsScan *s,
    uint32_t target)
{
    if (!s)
        return -1;

    for (uint32_t i = 0; i < s->n_dec; ++i)
    {
        if (s->decorations[i].target == target)
            return (int)i;
    }

    return -1;
}

/*
 * Returns true only for descriptors backed by upgraded stereo render
 * targets. Material textures, lookup tables and other resources remain
 * regular sampler2D objects.
 */
static bool
fs_binding_is_stereo_attachment(
const FsScan *s,
uint32_t var)
{
    int vi = fs_var_index(s, var);
    if (vi < 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_BINDING_LOOKUP_FAIL var=%u",
            var);
        return false;
    }
    const FsVariableInfo *v = &s->vars[vi];
    STEREO_LOG_VERBOSE(
        "FS_BINDING_INFO var=%u type=%u storage=%u set=%u binding=%u",
        v->id,
        v->type,
        v->storage,
        v->set,
        v->binding);
    STEREO_LOG_VERBOSE(
        "FS_BINDING_TYPE var=%u "
        "storage=%u "
        "type=%u "
        "sampledImage=%u "
        "n_img=%u",
        v->id,
        v->storage,
        v->type,
        s->n_img);
    /*
     * Input attachments are framebuffer attachments.
     */
    if (v->storage == SpvStorageClassInput)
    {
        STEREO_LOG_VERBOSE(
            "FS_BINDING_INPUT_ATTACHMENT var=%u stereo=1",
            var);
        return true;
    }
    /*
     * Deferred rendering attachments:
     *
     * binding 0 = depth/position
     * binding 1 = normal
     * binding 2 = albedo
     * binding 3 = specular
     * binding 4 = SSAO/deferred intermediate
     */
    bool stereo =
        (v->binding <= 4);
    STEREO_LOG_VERBOSE(
        "FS_BINDING_RESULT "
        "var=%u "
        "storage=%u "
        "set=%u "
        "binding=%u "
        "stereo=%u",
        var,
        v->storage,
        v->set,
        v->binding,
        stereo);
    return stereo;
}

static uint32_t fs_result_type_of(FsScan *s,
    const uint32_t *in,
    size_t in_c,
    uint32_t result_id)
{
    for (size_t i = 5; i < in_c;)
    {
        uint32_t wc = in[i] >> 16;
        uint32_t op = in[i] & 0xffff;
        if (!wc)
            break;
        if ((op == SpvOpCompositeConstruct ||
             op == SpvOpConstant ||
             op == SpvOpConstantComposite ||
             op == SpvOpCompositeExtract ||
             op == SpvOpVectorShuffle ||
             op == SpvOpLoad ||
             op == SpvOpAccessChain ||
             op == SpvOpCopyObject ||
             op == SpvOpBitcast ||
             op == SpvOpPhi ||
             op == SpvOpImageFetch) &&
            wc >= 3 &&
            in[i + 2] == result_id)
        {
            return in[i + 1];
        }
        i += wc;
    }
    return 0;
}

static bool
fs_should_patch_sample(
    const FsScan *s,
    uint64_t spv_hash,
    uint32_t descriptor_var)
{
    int vi = fs_var_index(s, descriptor_var);
    if (vi < 0)
        return false;
    uint32_t binding = s->vars[vi].binding;
    uint32_t set     = s->vars[vi].set;
    ///*
    // * SSAO noise is a mono lookup texture. In the SSAO generator shader
    // * (35d504ebec7cf2d7) it must not be arrayed or ViewIndex-shifted.
    // */
    //if (spv_hash == 0x35d504ebec7cf2d7ULL && binding == 2)
    //{
    //    STEREO_LOG_VERBOSE(
    //        "FS_SAMPLE_SKIP_NOISE "
    //        "hash=%016llx "
    //        "descriptor=%u "
    //        "set=%u "
    //        "binding=%u "
    //        "storage=%u "
    //        "type=%u",
    //        (unsigned long long)spv_hash,
    //        descriptor_var,
    //        set,
    //        binding,
    //        s->vars[vi].storage,
    //        s->vars[vi].type);
    //    STEREO_LOG_VERBOSE(
    //        "FS_NOISE_REASON "
    //        "hash=%016llx "
    //        "descriptor=%u "
    //        "set=%u "
    //        "binding=%u "
    //        "reason=SSAO_NOISE_BINDING2",
    //        (unsigned long long)spv_hash,
    //        descriptor_var,
    //        set,
    //        binding);
    //    return false;
    //}
    STEREO_LOG_VERBOSE(
        "FS_PATCH_DECISION "
        "hash=%016llx "
        "descriptor=%u "
        "set=%u "
        "binding=%u",
        (unsigned long long)spv_hash,
        descriptor_var,
        set,
        binding);
    return fs_binding_is_stereo_attachment(s, descriptor_var);
}

/*
 * Human-readable SPIR-V opcode names used only for diagnostics.
 * Keep this table small and focused on image/texture operations that
 * the fragment shader patcher cares about.
 */
static const char *
spv_op_name(uint32_t op)
{
    switch (op)
    {
    case SpvOpCopyObject:
        return "OpCopyObject";
    case SpvOpVariable:
        return "OpVariable";
    case SpvOpLoad:
        return "OpLoad";
    case SpvOpSampledImage:
        return "OpSampledImage";
    case SpvOpImage:
        return "OpImage";
    case SpvOpImageSampleImplicitLod:
        return "OpImageSampleImplicitLod";
    case SpvOpImageQuerySize:
        return "OpImageQuerySize";
    case SpvOpImageQuerySizeLod:
        return "OpImageQuerySizeLod";
    case SpvOpImageSampleExplicitLod:
        return "OpImageSampleExplicitLod";
    case SpvOpImageSampleDrefImplicitLod:
        return "OpImageSampleDrefImplicitLod";
    case SpvOpImageSampleDrefExplicitLod:
        return "OpImageSampleDrefExplicitLod";
    case SpvOpImageGather:
        return "OpImageGather";
    case SpvOpImageDrefGather:
        return "OpImageDrefGather";
    case SpvOpImageFetch:
        return "OpImageFetch";
    case SpvOpImageRead:
        return "OpImageRead";
    case SpvOpImageWrite:
        return "OpImageWrite";
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
    default:
        return "Unknown";
    }
}

static bool
fs_is_image_related_type(
    const FsScan *s,
    uint32_t type)
{
    if (!s)
        return false;

    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        if (s->images[i].id == type)
            return true;
    }

    for (uint32_t i = 0; i < s->n_si; ++i)
    {
        if (s->si_ids[i] == type)
            return true;
    }

    return false;
}

/*═══════════════════════════════════════════════════════════════════════
 * FsScan lookup helpers
 *
 * These provide a single implementation for common table lookups used
 * throughout the fragment shader parser.
 *═══════════════════════════════════════════════════════════════════════*/
static int
fs_find_function(
    const FsScan *s,
    uint32_t function_id)
{
    if (!s)
        return -1;
    for (uint32_t i = 0; i < s->n_function; ++i)
    {
        if (s->functions[i].id == function_id)
            return (int)i;
    }
    return -1;
}
static int
fs_find_load(
    const FsScan *s,
    uint32_t value_id)
{
    if (!s)
    {
        STEREO_LOG_VERBOSE(
            "FS_FIND_LOAD_MISS value=%u",
            value_id);
        return -1;
    }
    for (uint32_t i = 0; i < s->n_load; ++i)
    {
        if (s->loads[i].id == value_id)
        {
            return (int)i;
        }
    }
    STEREO_LOG_VERBOSE(
        "FS_FIND_LOAD_MISS value=%u",
        value_id);
    return -1;
}
static int
fs_find_parameter(
    const FsScan *s,
    uint32_t parameter_id)
{
    if (!s)
        return -1;
    for (uint32_t i = 0; i < s->n_param; ++i)
    {
        if (s->params[i].id == parameter_id)
            return (int)i;
    }
    return -1;
}
static int
fs_find_call_parameter(
    const FsScan *s,
    uint32_t parameter_id)
{
    if (!s)
        return -1;
    for (uint32_t i = 0; i < s->n_call; ++i)
    {
        if (s->calls[i].parameter_id == parameter_id)
            return (int)i;
    }
    return -1;
}
/*═══════════════════════════════════════════════════════════════════════
 * Ownership helpers
 *═══════════════════════════════════════════════════════════════════════*/
/*
 * Record ownership of an SSA value that ultimately represents an image,
 * sampled image, or image object.
 *
 * The mapping is:
 *
 *     SSA value  --->  descriptor variable
 *
 * If the SSA value is already known (for example after propagation through
 * OpImage or OpCopyObject), simply update the owner instead of creating a
 * duplicate entry.
 */
static void
fs_add_load_mapping(
    FsScan *s,
    uint32_t value_id,
    uint32_t owner)
{
    if (!s)
        return;
    int index =
        fs_find_load(
            s,
            value_id);
    if (index >= 0)
    {
        s->loads[index].owner_var = owner;
        STEREO_LOG_VERBOSE(
            "FS_LOAD_UPDATE id=%u owner=%u index=%d",
            value_id,
            owner,
            index);
        return;
    }
    if (s->n_load >= FS_MAX_LOADS)
    {
        STEREO_LOG_VERBOSE(
            "FS_LOAD_OVERFLOW id=%u",
            value_id);
        return;
    }
    FsLoadInfo *load =
        &s->loads[s->n_load++];
    memset(
        load,
        0,
        sizeof(*load));
    load->id =
        value_id;
    load->owner_var =
        owner;
    load->source_id =
        value_id;
    load->binding =
        0xffffffffu;
    STEREO_LOG_VERBOSE(
        "FS_LOAD_ADD id=%u owner=%u index=%u",
        value_id,
        owner,
        s->n_load - 1);
}
/*
 * Resolve the descriptor variable that owns a given SSA image value.
 *
 * The load table records ownership propagation through image-producing
 * instructions (OpLoad → OpImage → OpCopyObject → OpSampledImage, etc.).
 *
 * Returns true if ownership is known.
 */
static bool
fs_resolve_load_owner(
    const FsScan *s,
    uint32_t value_id,
    uint32_t *owner)
{
    if (!s)
        return false;
    int index =
        fs_find_load(
            s,
            value_id);
    if (index < 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_OWNER_LOOKUP_MISS value=%u",
            value_id);
        return false;
    }
    if (owner)
    {
        *owner =
            s->loads[index].owner_var;
    }
    STEREO_LOG_VERBOSE(
        "FS_OWNER_LOOKUP value=%u owner=%u index=%d",
        value_id,
        s->loads[index].owner_var,
        index);
    return true;
}
/*═══════════════════════════════════════════════════════════════════════
 * Instruction scanners
 *═══════════════════════════════════════════════════════════════════════*/

static uint32_t
fs_find_matching_sampled_image(
    const uint32_t *in,
    size_t in_c,
    uint32_t image_type)
{
    for (size_t i = 5; i < in_c;)
    {
        uint32_t wc = in[i] >> 16;
        uint32_t op = in[i] & 0xffff;
        if (!wc || i + wc > in_c)
            break;
        if (op == SpvOpTypeSampledImage &&
            wc >= 3 &&
            in[i + 2] == image_type)
        {
            return in[i + 1];
        }
        i += wc;
    }
    return 0;
}
static uint32_t
fs_find_matching_image_type(
    const uint32_t *in,
    size_t in_c,
    uint32_t sampled_type,
    uint32_t dim,
    uint32_t depth,
    uint32_t arrayed,
    uint32_t ms,
    uint32_t sampled,
    uint32_t format)
{
    for (size_t i = 5; i < in_c;)
    {
        uint32_t wc = in[i] >> 16;
        uint32_t op = in[i] & 0xffff;
        if (!wc || i + wc > in_c)
            break;
        if (op == SpvOpTypeImage &&
            wc >= 9 &&
            in[i + 2] == sampled_type &&
            in[i + 3] == dim &&
            in[i + 4] == depth &&
            in[i + 5] == arrayed &&
            in[i + 6] == ms &&
            in[i + 7] == sampled &&
            in[i + 8] == format)
        {
            return in[i + 1];
        }
        i += wc;
    }
    return 0;
}

static int
fs_find_matching_array_image(FsScan *s, const FsImageInfo *src)
{
    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        const FsImageInfo *img = &s->images[i];
        STEREO_LOG_VERBOSE(
            "FS_MATCH_CHECK cur=%u cand=%u "
            "sampledImage=%u/%u dim=%u/%u depth=%u/%u "
            "arr=%u/%u ms=%u/%u sampled=%u/%u fmt=%u/%u",
            src->id,
            img->id,
            src->sampled_type, img->sampled_type,
            src->dim,          img->dim,
            src->depth,        img->depth,
            src->arrayed,      img->arrayed,
            src->ms,           img->ms,
            src->sampled,      img->sampled,
            src->format,       img->format);
        /* OpTypeSampledImage wrappers may differ while the underlying
         * OpTypeImage declarations are identical. Ignore wrapper ids. */
        if (img->id           == src->id)           continue;
        if (img->dim          != src->dim)          continue;
        if (img->depth        != src->depth)        continue;
        if (img->ms           != src->ms)           continue;
        if (img->sampled      != src->sampled)      continue;
        if (img->format       != src->format)       continue;
        if (img->arrayed == 1)
        {
            STEREO_LOG_VERBOSE(
                "FS_MATCH_FOUND src=%u reuse=%u",
                src->id,
                img->id);
            return (int)i;
        }
    }
    return -1;
}

static int
fs_find_image_by_sampled_image(
    FsScan *s,
    uint32_t sampled_image_type)
{
    STEREO_LOG_VERBOSE(
        "FS_FIND_IMAGE_BY_SAMPLE target=%u n_img=%u",
        sampled_image_type,
        s->n_img);
    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        STEREO_LOG_VERBOSE(
            "FS_FIND_IMAGE_BY_SAMPLE_ENTRY idx=%u img=%u sampled=%u sampled_id=%u",
            i,
            s->images[i].id,
            s->images[i].sampled_type,
            s->images[i].sampled_type_id);
        STEREO_LOG_VERBOSE(
            "FS_FIND_COMPARE target=%u sampled=%u sampled_id=%u",
            sampled_image_type,
            s->images[i].sampled_type,
            s->images[i].sampled_type_id);
        if (s->images[i].sampled_type_id == sampled_image_type)
            return (int)i;
    }
    return -1;
}

static int
fs_find_image_by_owner(
    FsScan *s,
    uint32_t owner_var)
{
    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        if (s->images[i].owner_var == owner_var)
            return (int)i;
    }
    return -1;
}

static void
fs_scan_type_instruction(
    FsScan *s,
    const uint32_t *ins,
    uint32_t op,
    uint32_t wc)
{
    if (!s || !ins)
        return;
    //if (wc >= 2)
    //{
    //    STEREO_LOG_VERBOSE(
    //        "FS_TYPE_DECL id=%u opcode=%u",
    //        ins[1],
    //        op);
    //}
    switch (op)
    {
    case SpvOpTypeFloat:
        if (wc >= 3 && ins[2] == 32)
        {
            s->float_id = ins[1];
        }
        break;
    case SpvOpTypeInt:
        if (wc >= 4 &&
            ins[2] == 32)
        {
            if (ins[3] == 1)
                s->int_id = ins[1];
            else
                s->uint_id = ins[1];
            STEREO_LOG_VERBOSE(
                "FS_TYPE_INT_DECL id=%u signed=%u int=%u uint=%u",
                ins[1],
                ins[3],
                s->int_id,
                s->uint_id);
        }
        break;
    case SpvOpTypeVector:
        if (wc >= 4)
        {
            if (ins[2] == s->int_id)
            {
                if (ins[3] == 2)
                    s->v2int_id = ins[1];
                else if (ins[3] == 3)
                    s->v3int_id = ins[1];
            }
            else if (ins[2] == s->uint_id)
            {
                if (ins[3] == 2)
                    s->v2uint_id = ins[1];
                else if (ins[3] == 3)
                    s->v3uint_id = ins[1];
            }
            else if (ins[2] == s->float_id &&
                     ins[3] == 3)
            {
                s->v3float_id = ins[1];
            }
        }
        break;
    case SpvOpTypeImage:
    {
        STEREO_LOG_VERBOSE(
            "FS_SCAN_TYPEIMAGE id=%u sampledType=%u dim=%u depth=%u arrayed=%u ms=%u sampled=%u format=%u",
            (wc >= 2) ? ins[1] : 0,
            (wc >= 3) ? ins[2] : 0,
            (wc >= 4) ? ins[3] : 0,
            (wc >= 5) ? ins[4] : 0,
            (wc >= 6) ? ins[5] : 0,
            (wc >= 7) ? ins[6] : 0,
            (wc >= 8) ? ins[7] : 0,
            (wc >= 9) ? ins[8] : 0);
        if (wc < 9)
            break;
        uint32_t type_id      = ins[1];
        uint32_t sampled_type = ins[2];
        uint32_t dim          = ins[3];
        uint32_t depth        = ins[4];
        uint32_t arrayed      = ins[5];
        uint32_t ms           = ins[6];
        uint32_t sampled      = ins[7];
        uint32_t format       = ins[8];
        if (dim == SpvDim2D &&
            s->n_img < FS_MAX_IMG)
        {
            STEREO_LOG_VERBOSE(
                "FS_NEW_IMAGE_SCAN "
                "idx=%u "
                "id=%u "
                "sampled=%u "
                "dim=%u "
                "arrayed=%u",
                s->n_img,
                type_id,
                sampled_type,
                dim,
                arrayed);
            FsImageInfo *img =
                &s->images[s->n_img++];
            STEREO_LOG_VERBOSE(
                "FS_IMAGE_NEW idx=%u imageType=%u n_img=%u",
                s->n_img - 1,
                type_id,
                s->n_img);
            memset(img, 0, sizeof(*img));
            img->id               = type_id;
            img->sampled_type     = sampled_type;
            img->dim              = dim;
            img->depth            = depth;
            img->arrayed          = arrayed;
            img->ms               = ms;
            img->sampled          = sampled;
            img->format           = format;
            img->patchable        = (arrayed == 0);
            img->stereo           = (arrayed != 0);
            img->replacement_type = 0;
            STEREO_LOG_VERBOSE(
                "FS_NEW_IMAGE_DONE "
                "idx=%u "
                "id=%u",
                s->n_img - 1,
                img->id);
            STEREO_LOG_VERBOSE(
                "FS_ARRAY_TYPE_PATCH "
                "imageType=%u "
                "sampledType=%u "
                "arrayed_before=%u "
                "arrayed_after=%u",
                type_id,
                sampled_type,
                arrayed,
                1u);
            STEREO_LOG_VERBOSE(
                "FS_BEFORE_ADD_LOG n_img=%u ptr=%p",
                s->n_img,
                (void *)s);
            STEREO_LOG_VERBOSE(
                "FS_ADD_IMAGE idx=%u n_img=%u id=%u ptr=%p",
                s->n_img - 1,
                s->n_img,
                img->id,
                (void *)s);
        }
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_TABLE_SIZE n_img=%u",
            s->n_img);
        break;
    }
    case SpvOpTypeSampledImage:
    {
        STEREO_LOG_VERBOSE(
            "FS_SAMPLED_BEGIN sampledType=%u imageType=%u n_img=%u",
            (wc >= 2) ? ins[1] : 0,
            (wc >= 3) ? ins[2] : 0,
            s->n_img);
        if (wc < 3)
            break;
        uint32_t sampled_image_id = ins[1];
        uint32_t image_type_id    = ins[2];
        STEREO_LOG_VERBOSE(
            "FS_TYPE_SAMPLED_IMAGE id=%u imageType=%u",
            sampled_image_id,
            image_type_id);
        for (uint32_t ii = 0; ii < s->n_img; ++ii)
        {
            STEREO_LOG_VERBOSE(
                "FS_SAMPLED_COMPARE idx=%u imageType=%u wanted=%u",
                ii,
                s->images[ii].id,
                image_type_id);
            if (s->images[ii].id == image_type_id)
            {
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLED_STORE idx=%u image=%u sampled=%u",
                    ii,
                    s->images[ii].id,
                    s->images[ii].sampled_type);
                /*
                 * Keep sampled_type as the OpTypeImage component type (%float, etc.).
                 * Store the OpTypeSampledImage wrapper separately.
                 */
                s->images[ii].sampled_type_id = sampled_image_id;
                STEREO_LOG_VERBOSE(
                    "FS_IMAGE_TYPE_BIND "
                    "idx=%u "
                    "imageType=%u "
                    "sampledType=%u",
                    ii,
                    image_type_id,
                    sampled_image_id);
                STEREO_LOG_VERBOSE(
                    "FS_TYPE_SAMPLED_IMAGE_MAP imageType=%u sampledImage=%u",
                    image_type_id,
                    sampled_image_id);
                break;
            }
        }
        if (s->n_si < FS_MAX_SI)
            s->si_ids[s->n_si++] = sampled_image_id;
        break;
    }
    case SpvOpTypePointer:
        for (uint32_t img = 0; img < s->n_img; ++img)
        {
            STEREO_LOG_VERBOSE(
                "FS_BEFORE_PTR idx=%u image=%u sampled=%u",
                img,
                s->images[img].id,
                s->images[img].sampled_type);
        }
        STEREO_LOG_VERBOSE(
            "FS_TYPE_POINTER id=%u storage=%u target=%u",
            (wc >= 2) ? ins[1] : 0,
            (wc >= 3) ? ins[2] : 0,
            (wc >= 4) ? ins[3] : 0);
        if (wc >= 4 &&
            ins[2] == SpvStorageClassInput &&
            s->int_id &&
            ins[3] == s->int_id)
        {
            s->ptr_int_in_id = ins[1];
            STEREO_LOG_VERBOSE(
                "FS_PTR_INT_INPUT id=%u",
                s->ptr_int_in_id);
        }
        if (wc >= 4)
        {
            for (uint32_t img = 0; img < s->n_img; ++img)
            {
                STEREO_LOG_VERBOSE(
                    "FS_PTR_COMPARE "
                    "idx=%u "
                    "image=%u "
                    "sampled_type=%u "
                    "sampled_type_id=%u "
                    "ptrTarget=%u",
                    img,
                    s->images[img].id,
                    s->images[img].sampled_type,
                    s->images[img].sampled_type_id,
                    ins[3]);
                if (s->images[img].sampled_type_id == ins[3])
                {
                    s->images[img].pointer_type = ins[1];
                    STEREO_LOG_VERBOSE(
                        "FS_POINTER_BIND "
                        "idx=%u "
                        "imageType=%u "
                        "sampledType=%u "
                        "pointerType=%u",
                        img,
                        s->images[img].id,
                        s->images[img].sampled_type,
                        ins[1]);
                    STEREO_LOG_VERBOSE(
                        "FS_IMAGE_POINTER image=%u sampled=%u pointer=%u",
                        s->images[img].id,
                        s->images[img].sampled_type,
                        s->images[img].pointer_type);
                    break;
                }
            }
        }
        break;
    default:
        break;
    }
}
/*
 * Process OpDecorate instructions.
 *
 * Descriptor decorations may appear before the corresponding
 * OpVariable declaration, so we cache them here and apply them
 * later when the variable is encountered.
 *
 * Handles:
 *   - Location
 *   - BuiltIn ViewIndex
 *   - Descriptor Binding
 *   - Descriptor Set
 *
 * Keeping this separate from fs_prescan() is important because
 * future projection-matrix handling will also need clean access
 * to descriptor metadata without depending on scan order.
 */
static void
fs_process_decoration(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (!s || !ins || wc < 4)
        return;

    uint32_t target     = ins[1];
    uint32_t decoration = ins[2];
    uint32_t value      = ins[3];

    if (target == 15)
    {
        STEREO_LOG_VERBOSE(
            "FS_DECORATION_VAR15 decoration=%u value=%u",
            decoration,
            value);
    }

    if (decoration == SpvDecorationBuiltIn &&
        value == SpvBuiltInViewIndex)
    {
        s->vi_var_id = target;
        STEREO_LOG_VERBOSE(
            "FS_VIEWINDEX_FOUND id=%u",
            target);
        return;
    }

    if (decoration == SpvDecorationLocation)
    {
        int index = fs_var_index(s, target);
        if (index >= 0)
        {
            s->vars[index].location = value;
            STEREO_LOG_VERBOSE(
                "FS_LOCATION_APPLY var=%u location=%u",
                target,
                value);
        }
        return;
    }

    if (decoration != SpvDecorationBinding &&
        decoration != SpvDecorationDescriptorSet)
    {
        return;
    }

    int index = -1;
    for (uint32_t i = 0; i < s->n_dec; ++i)
    {
        if (s->decorations[i].target == target)
        {
            index = (int)i;
            break;
        }
    }

    if (index < 0)
    {
        if (s->n_dec >= FS_MAX_VARS)
        {
            STEREO_LOG_VERBOSE(
                "FS_DECORATION_OVERFLOW target=%u",
                target);
            return;
        }

        index = (int)s->n_dec++;
        FsDecorationInfo *dec = &s->decorations[index];
        memset(dec, 0, sizeof(*dec));
        dec->target   = target;
        dec->set      = 0xffffffffu;
        dec->binding  = 0xffffffffu;
        dec->location = 0xffffffffu;
    }

    FsDecorationInfo *dec = &s->decorations[index];
    if (decoration == SpvDecorationBinding)
        dec->binding = value;
    else
        dec->set = value;

    int var_index = fs_var_index(s, target);
    if (var_index >= 0)
    {
        FsVariableInfo *var = &s->vars[var_index];
        if (decoration == SpvDecorationBinding)
            var->binding = value;
        else
            var->set = value;

        if (var->storage == SpvStorageClassUniform &&
            var->binding == 4u)
        {
            var->is_projection_ubo = true;
            STEREO_LOG_VERBOSE(
                "FS_PROJECTION_UBO_DECORATED var=%u set=%u binding=%u",
                var->id,
                var->set,
                var->binding);
        }
    }
}
/*═══════════════════════════════════════════════════════════════════════
 * Pass 2: Descriptor variables and decorations.
 *
 * Decorations (Binding, DescriptorSet, BuiltIn, Location) may legally
 * appear either before or after OpVariable, so this pass maintains a
 * temporary decoration cache which is applied whenever the matching
 * variable declaration is encountered.
 *
 * This pass does NOT determine whether a descriptor is stereo.
 * That decision happens later after image ownership is known.
 *═══════════════════════════════════════════════════════════════════════*/
static void
fs_scan_variable_instruction(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (!s || !ins || wc < 4)
        return;
    if (s->n_var >= FS_MAX_VARS)
    {
        STEREO_LOG_VERBOSE(
            "FS_VAR_OVERFLOW id=%u",
            ins[2]);
        return;
    }
    FsVariableInfo *var = &s->vars[s->n_var++];
    memset(var, 0, sizeof(*var));
    var->id       = ins[2];
    var->type     = ins[1];
    var->storage  = ins[3];
    var->set      = 0xffffffffu;
    var->binding  = 0xffffffffu;
    var->location = 0xffffffffu;
    var->is_projection_ubo = false;
    /*
     * Decorations may legally appear before OpVariable.
     * Apply cached DescriptorSet, Binding, and Location values now.
     */
    for (uint32_t i = 0; i < s->n_dec; ++i)
    {
        FsDecorationInfo *dec = &s->decorations[i];
        if (dec->target != var->id)
            continue;
        if (dec->set != 0xffffffffu)
            var->set = dec->set;
        if (dec->binding != 0xffffffffu)
            var->binding = dec->binding;
        if (dec->location != 0xffffffffu)
            var->location = dec->location;
        STEREO_LOG_VERBOSE(
            "FS_REGISTER_VAR "
            "id=%u "
            "type=%u "
            "storage=%u "
            "set=%u "
            "binding=%u",
            var->id,
            var->type,
            var->storage,
            var->set,
            var->binding);
    }
    STEREO_LOG_VERBOSE(
        "FS_DESCRIPTOR_CREATE "
        "var=%u "
        "type=%u "
        "storage=%u "
        "set=%u "
        "binding=%u",
        var->id,
        var->type,
        var->storage,
        var->set,
        var->binding);
    STEREO_LOG_VERBOSE(
        "FS_VAR_TYPE_LOOKUP "
        "var=%u "
        "type=%u",
        var->id,
        var->type);

    for (uint32_t ii = 0; ii < s->n_img; ++ii)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_TYPE "
            "idx=%u "
            "id=%u "
            "sampledType=%u "
            "arrayed=%u",
            ii,
            s->images[ii].id,
            s->images[ii].sampled_type,
            s->images[ii].arrayed);
    }
    if (var->storage == SpvStorageClassUniformConstant)
    {
        for (uint32_t ii = 0; ii < s->n_img; ++ii)
        {
            if (s->images[ii].id == var->type)
            {
                STEREO_LOG_VERBOSE(
                    "FS_TYPE_IMAGE "
                    "var=%u "
                    "imageType=%u "
                    "sampledType=%u "
                    "dim=%u "
                    "arrayed=%u "
                    "set=%u "
                    "binding=%u",
                    var->id,
                    s->images[ii].id,
                    s->images[ii].sampled_type,
                    s->images[ii].dim,
                    s->images[ii].arrayed,
                    var->set,
                    var->binding);
                break;
            }
        }
    }
    if (var->storage == SpvStorageClassUniform &&
        var->binding == 4u)
    {
        var->is_projection_ubo = true;
        STEREO_LOG_VERBOSE(
            "FS_PROJECTION_UBO var=%u type=%u set=%u binding=%u",
            var->id,
            var->type,
            var->set,
            var->binding);
    }
    if (var->id == 15)
    {
        STEREO_LOG_VERBOSE(
            "FS_DEBUG_VAR15 type=%u storage=%u set=%u binding=%u location=%u",
            var->type,
            var->storage,
            var->set,
            var->binding,
            var->location);
    }
    if (var->storage == SpvStorageClassUniform ||
        var->storage == SpvStorageClassUniformConstant)
    {
        STEREO_LOG_VERBOSE(
            "FS_DESCRIPTOR_VAR id=%u type=%u storage=%u set=%u binding=%u",
            var->id,
            var->type,
            var->storage,
            var->set,
            var->binding);
    }
}
/*
 * Register an OpVariable instruction.
 *
 * Variables are the bridge between:
 *
 *     sampled image objects
 *             |
 *             v
 *     descriptor variables
 *             |
 *             v
 *     descriptor set / binding
 *
 * Decorations may legally appear before OpVariable, so after
 * registering the variable we apply any cached metadata from
 * the decoration table.
 *
 * This function intentionally does NOT classify stereo resources.
 * Classification belongs later, after image provenance has been
 * resolved.
 */
static void
fs_scan_function_parameter(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (wc < 3)
        return;
    if (!s->in_function)
    {
        STEREO_LOG_VERBOSE(
            "FS_PARAM_OUTSIDE_FUNCTION id=%u",
            ins[2]);
        return;
    }
    if (s->n_param >= FS_MAX_PARAMS)
    {
        STEREO_LOG_VERBOSE(
            "FS_PARAM_OVERFLOW id=%u",
            ins[2]);
        return;
    }
    uint32_t idx =
        s->n_param++;
    s->params[idx].id =
        ins[2];
    s->params[idx].type =
        ins[1];
    s->params[idx].function_id =
        s->current_function_id;
    s->params[idx].index =
        s->current_param_index;
    STEREO_LOG_VERBOSE(
        "FS_PARAM_ADD function=%u index=%u id=%u type=%u",
        s->current_function_id,
        s->current_param_index,
        ins[2],
        ins[1]);
    /*
     * Parameters do not own descriptors directly.
     *
     * Example:
     *
     *   OpFunctionParameter %image %param
     *
     * The actual descriptor ownership is resolved later through:
     *
     *   OpFunctionCall arguments
     *          |
     *          v
     *   parameter id
     *          |
     *          v
     *   originating descriptor variable
     *
     * This deferred resolution is required for deferred renderers
     * where image sampling happens inside helper functions.
     */
    s->current_param_index++;
}
/*
 * Scan function structure.
 *
 * Responsibilities:
 *  - track current function
 *  - remember first function offset
 *  - record every function parameter
 *  - build lookup tables used later to resolve descriptor ownership
 *
 * This performs no image analysis; it only records function metadata.
 */
static void
fs_scan_function(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (!s || !ins || wc < 3)
        return;
    uint32_t function_id =
        ins[2];
    /*
     * Track current function context while scanning.
     *
     * SPIR-V functions can contain parameters that appear before
     * the actual image operations using them.  We therefore record
     * function ownership first, then resolve argument forwarding
     * later in fs_fixup_function_parameters().
     */
    s->in_function = true;
    s->current_function_id = function_id;
    s->current_param_index = 0;
    if (s->n_function >= FS_MAX_FUNCTIONS)
    {
        STEREO_LOG_VERBOSE(
            "FS_FUNCTION_OVERFLOW id=%u",
            function_id);
        return;
    }
    FsFunctionInfo *fn =
        &s->functions[s->n_function++];
    fn->id =
        function_id;
    fn->first_param =
        s->n_param;
    fn->type_id =
        (wc >= 5) ? ins[4] : 0;
    STEREO_LOG_VERBOSE(
        "FS_FUNCTION_REGISTER id=%u index=%u firstParam=%u type=%u",
        function_id,
        s->n_function - 1,
        fn->first_param,
        fn->type_id);
    STEREO_LOG_VERBOSE(
        "FS_FUNCTION_BEGIN id=%u",
        function_id);
}
/*
 * Track OpLoad instructions that produce image-related objects.
 *
 * SPIR-V image usage commonly looks like:
 *
 *     OpLoad          %image   %descriptor
 *     OpSampledImage  %sampled %image %sampler
 *     OpImageSample   ...
 *
 * We cannot classify the image immediately because:
 *
 *   - descriptor decorations may appear later
 *   - function parameters may hide the originating variable
 *   - image objects can be copied through intermediate IDs
 *
 * Therefore this function only records provenance.
 *
 * Later passes resolve:
 *   image ID -> descriptor variable -> set/binding
 */
static void
fs_scan_load_instruction(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (!s || !ins || wc < 4)
        return;

    uint32_t result_type = ins[1];
    uint32_t result_id   = ins[2];
    uint32_t source_id   = ins[3];
    STEREO_LOG_VERBOSE(
        "FS_LOAD_SOURCE result=%u variable=%u",
        ins[2],
        source_id);
    STEREO_LOG_VERBOSE(
        "FS_LOAD_INPUT result=%u source=%u type=%u",
        result_id,
        source_id,
        result_type);
    uint32_t owner = 0;
    bool have_owner = fs_resolve_load_owner(s, source_id, &owner);

    if (!have_owner)
    {
        /*
         * The source may be a function parameter or another SSA value
         * that will be fixed up later.
         */
        owner = source_id;
        STEREO_LOG_VERBOSE(
            "FS_LOAD_DEFERRED result=%u source=%u",
            result_id,
            source_id);
    }

    STEREO_LOG_VERBOSE(
        "FS_LOAD_OWNER_FINAL result=%u source=%u owner=%u resolved=%u",
        result_id,
        source_id,
        owner,
        have_owner);
    int owner_var_index = fs_var_index(s, owner);
    bool from_projection_ubo =
        (owner_var_index >= 0 &&
         s->vars[owner_var_index].is_projection_ubo);

    bool image_related =
        fs_is_image_related_type(s, result_type);

    /*
     * Keep tracking normal image-related loads as before.
     * Also keep projection UBO loads even when they are not image-related,
     * because the FS uses them for convergence/projection reconstruction.
     */
    if (!image_related && !from_projection_ubo)
        return;

    if (s->n_load >= FS_MAX_LOADS)
    {
        STEREO_LOG_VERBOSE(
            "FS_LOAD_OVERFLOW result=%u",
            result_id);
        return;
    }

    FsLoadInfo *li = &s->loads[s->n_load++];
    memset(li, 0, sizeof(*li));

    li->id            = result_id;
    li->source_id     = source_id;
    li->owner_var     = owner;
    li->binding       = 0xffffffffu;
    li->from_projection = from_projection_ubo;
    li->from_view       = false;

    if (owner_var_index >= 0)
    {
        li->binding = s->vars[owner_var_index].binding;
    }

    int src_var = fs_var_index(s, source_id);
    if (src_var >= 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_LOAD_SOURCE result=%u sourceVar=%u set=%u binding=%u type=%u proj=%u",
            result_id,
            source_id,
            s->vars[src_var].set,
            s->vars[src_var].binding,
            s->vars[src_var].type,
            li->from_projection);
    }
    else
    {
        STEREO_LOG_VERBOSE(
            "FS_LOAD_SOURCE_UNKNOWN result=%u source=%u",
            result_id,
            source_id);
    }

    if (from_projection_ubo)
    {
        STEREO_LOG_VERBOSE(
            "FS_PROJECTION_LOAD result=%u owner=%u set=%u binding=%u type=%u",
            result_id,
            owner,
            (owner_var_index >= 0) ? s->vars[owner_var_index].set : 0xffffffffu,
            (owner_var_index >= 0) ? s->vars[owner_var_index].binding : 0xffffffffu,
            (owner_var_index >= 0) ? s->vars[owner_var_index].type : 0xffffffffu);
    }

    STEREO_LOG_VERBOSE(
        "FS_LOAD_REGISTER result=%u owner=%u type=%u proj=%u",
        result_id,
        owner,
        result_type,
        li->from_projection);
}
/*
 * Track OpFunctionCall relationships.
 *
 * SPIR-V functions make descriptor ownership difficult because
 * a sampled image may flow through parameters:
 *
 *   main()
 *      |
 *      | OpFunctionCall
 *      v
 *   helper(image)
 *      |
 *      | OpFunctionParameter
 *      v
 *   OpLoad
 *
 * During the first scan we do not yet know every parameter owner.
 *
 * Therefore this function records:
 *
 *   function ID
 *   argument index
 *   argument value
 *
 * Later fs_fixup_function_parameters() resolves:
 *
 *   parameter ID -> original descriptor variable
 *
 * This separation keeps resource classification independent
 * from SPIR-V function ordering.
 */
static void
fs_scan_function_call(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (!s || !ins || wc < 4)
        return;
    uint32_t result_id =
        ins[2];
    uint32_t function_id =
        ins[3];
    uint32_t argument_count =
        wc - 4;
    STEREO_LOG_VERBOSE(
        "FS_FUNCTION_CALL result=%u function=%u argc=%u",
        result_id,
        function_id,
        argument_count);
    for (uint32_t arg = 0;
         arg < argument_count;
         ++arg)
    {
        uint32_t value =
            ins[4 + arg];
        STEREO_LOG_VERBOSE(
            "FS_FUNCTION_ARG index=%u value=%u",
            arg,
            value);
        if (s->n_call >= FS_MAX_CALLS)
        {
            STEREO_LOG_VERBOSE(
                "FS_CALL_OVERFLOW function=%u arg=%u",
                function_id,
                arg);
            continue;
        }
        FsCallInfo *call =
            &s->calls[s->n_call++];
        memset(
            call,
            0,
            sizeof(*call));
        call->function_id =
            function_id;
        /*
         * During the first scan this is the argument position.
         * fs_fixup_function_parameters() converts it into the
         * real parameter ID after the function table is known.
         */
        call->parameter_index =
            arg;
        call->argument_var =
            value;
        call->parameter_id =
            0;
        STEREO_LOG_VERBOSE(
            "FS_CALL_STORE function=%u argIndex=%u value=%u total=%u",
            function_id,
            arg,
            value,
            s->n_call);
    }
}
/*
 * Ownership propagation
 */
/*
 * Propagate descriptor ownership through image-producing instructions.
 *
 * Many deferred renderers perform chains such as:
 *
 *      OpLoad
 *          ↓
 *      OpImage
 *          ↓
 *      OpCopyObject
 *          ↓
 *      OpImageSample*
 *
 * Each intermediate SSA value must retain the descriptor ownership of the
 * original OpLoad so later sampling instructions can still recover the
 * descriptor binding.
 */
static void
fs_track_image_propagation(
    FsScan *s,
    const uint32_t *ins,
    uint32_t op,
    uint32_t wc)
{
    if (!s || wc < 4)
        return;
    bool propagate =
        (op == SpvOpImage) ||
        (op == SpvOpCopyObject);
    if (!propagate)
        return;
    uint32_t result_id = ins[2];
    uint32_t source_id = ins[3];
    int src =
        fs_find_load(
            s,
            source_id);
    if (src < 0)
    {
        return;
    }
    STEREO_LOG_VERBOSE(
        "FS_PROP_IMAGE op=%s result=%u source=%u load=%d owner=%u binding=%u sourceOwner=%u",
        spv_op_name(op),
        result_id,
        source_id,
        src,
        s->loads[src].owner_var,
        s->loads[src].binding,
        s->loads[src].source_id);
    if (s->n_load >= FS_MAX_LOADS)
    {
        STEREO_LOG_VERBOSE(
            "FS_PROP_OVERFLOW result=%u",
            result_id);
        return;
    }
    FsLoadInfo *dst =
        &s->loads[s->n_load++];
    *dst = s->loads[src];
    dst->id = result_id;
    STEREO_LOG_VERBOSE(
        "FS_PROPAGATE op=%s src=%u dst=%u owner=%u source=%u binding=%u",
        spv_op_name(op),
        source_id,
        result_id,
        dst->owner_var,
        dst->source_id,
        dst->binding);
}
/*
 * Track OpSampledImage ownership.
 *
 * OpSampledImage combines:
 *
 *      image object
 *          +
 *      sampler object
 *
 * into a sampled-image object consumed by OpImageSample*.
 *
 * We only care about preserving the descriptor ownership of the image
 * object so later texture sampling instructions can still recover the
 * originating descriptor binding.
 */
static void
fs_track_sampled_image(
    FsScan *s,
    const uint32_t *ins,
    uint32_t wc)
{
    if (!s || wc < 5)
        return;
    /*
     * Ignore non-sampled-image result types.
     */
    if (!fs_id_in(
            s->si_ids,
            s->n_si,
            ins[1]))
    {
        STEREO_LOG_VERBOSE(
            "FS_SAMPLED_IMAGE_SKIP resultType=%u result=%u image=%u sampler=%u n_si=%u",
            ins[1],
            (wc >= 3) ? ins[2] : 0,
            (wc >= 4) ? ins[3] : 0,
            (wc >= 5) ? ins[4] : 0,
            s->n_si);
        return;
    }
    uint32_t result_id  = ins[2];
    uint32_t image_id   = ins[3];
    uint32_t sampler_id = ins[4];
    int src =
        fs_find_load(
            s,
            image_id);
    if (src < 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_SAMPLED_IMAGE_NO_SOURCE result=%u image=%u sampler=%u",
            result_id,
            image_id,
            sampler_id);
        return;
    }
    if (s->n_load >= FS_MAX_LOADS)
    {
        STEREO_LOG_VERBOSE(
            "FS_SAMPLED_IMAGE_OVERFLOW result=%u",
            result_id);
        return;
    }
    FsLoadInfo *dst =
        &s->loads[s->n_load++];
    *dst = s->loads[src];
    dst->id = result_id;
    STEREO_LOG_VERBOSE(
        "FS_LOAD_REGISTER "
        "id=%u "
        "source=%u "
        "owner=%u "
        "binding=%u",
        dst->id,
        dst->source_id,
        dst->owner_var,
        dst->binding);
    STEREO_LOG_VERBOSE(
        "FS_SAMPLED_IMAGE_REGISTER result=%u image=%u owner=%u binding=%u",
        result_id,
        image_id,
        dst->owner_var,
        dst->binding);
    STEREO_LOG_VERBOSE(
        "FS_SAMPLED_IMAGE result=%u image=%u sampler=%u owner=%u source=%u",
        result_id,
        image_id,
        sampler_id,
        dst->owner_var,
        dst->source_id);
}
/*
 * Scan image sampling/fetch/read/write instructions.
 *
 * At this point we do not modify these instructions. The goal is to:
 *
 *   • determine which descriptor is being sampled
 *   • log the descriptor binding
 *   • verify ownership propagation worked correctly
 *
 * The actual SPIR-V rewriting happens later in the patch pass.
 */
static void
fs_scan_image_operation(
    FsScan *s,
    const uint32_t *ins,
    uint32_t op,
    uint32_t wc)
{
    STEREO_LOG_VERBOSE(
        "FS_IMAGE_SCAN op=%s imageOperand=%u resultType=%u result=%u",
        spv_op_name(op),
        (wc >= 4) ? ins[3] : 0,
        (wc >= 2) ? ins[1] : 0,
        (wc >= 3) ? ins[2] : 0);
    if (!s || wc < 5)
        return;
    switch (op)
    {
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageFetch:
    case SpvOpImageRead:
    case SpvOpImageWrite:
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_WRITE image=%u value=%u",
            ins[3],
            ins[4]);
        break;
    default:
        return;
    }
    uint32_t image_id = ins[3];
    int load =
        fs_find_load(
            s,
            image_id);
    if (load < 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_NO_LOAD image=%u op=%s",
            image_id,
            spv_op_name(op));
        return;
    }
    FsLoadInfo *li =
        &s->loads[load];
    if (li->owner_var == 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_UNRESOLVED image=%u source=%u",
            image_id,
            li->source_id);
        return;
    }
    int var =
        fs_var_index(
            s,
            li->owner_var);
    if (var < 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_OWNER_UNKNOWN owner=%u",
            li->owner_var);
        return;
    }
    bool stereo =
        fs_binding_is_stereo_attachment(
            s,
            li->owner_var);
    STEREO_LOG_VERBOSE(
        "FS_SAMPLE_CLASSIFIED image=%u owner=%u binding=%u stereo=%u op=%s",
        image_id,
        li->owner_var,
        s->vars[var].binding,
        stereo,
        spv_op_name(op));
    li->binding =
        s->vars[var].binding;
    STEREO_LOG_VERBOSE(
        "FS_IMAGE_SAMPLE op=%s image=%u owner=%u set=%u binding=%u stereo=%u proj=%u view=%u",
        spv_op_name(op),
        image_id,
        li->owner_var,
        s->vars[var].set,
        s->vars[var].binding,
        stereo,
        li->from_projection,
        li->from_view);
    if (stereo)
    {
        STEREO_LOG_VERBOSE(
            "FS_STEREO_RESOURCE image=%u binding=%u proj=%u",
            image_id,
            s->vars[var].binding,
            li->from_projection);
    }
    STEREO_LOG_VERBOSE(
        "FS_IMAGE_OWNER image=%u owner=%u",
        (wc >= 4) ? ins[3] : 0,
        li->owner_var);
}
/*
 * Instruction dispatchers
 * ----------------------
 * Module traversal lives here. The individual semantic handlers above should
 * never walk the SPIR-V stream themselves.
 */
/*
 * Scan one SPIR-V instruction.
 *
 * This dispatcher performs the semantic analysis pass used by the
 * fullscreen-fragment patcher. Each instruction category is handled by a
 * dedicated helper so fs_prescan() only performs module traversal.
 *
 * No SPIR-V is modified here; this pass only records metadata needed by the
 * later patching phase.
 */
static void
fs_scan_instruction(
    FsScan *s,
    const uint32_t *ins,
    uint32_t op,
    uint32_t wc)
{
    STEREO_LOG_VERBOSE(
        "FS_SCAN_STATE_BEGIN op=%s n_img=%u",
        spv_op_name(op),
        s ? s->n_img : 999u);
    if (!s || !ins)
        return;
    STEREO_LOG_VERBOSE(
        "FS_SCAN op=%s(%u) wc=%u n_img=%u",
        spv_op_name(op),
        op,
        wc,
        s->n_img);
    /*
     * Diagnostic: log every image operation encountered during
     * the prescan so we know exactly which SPIR-V instructions
     * this shader uses for MSAA resolve/final composition.
     */
    switch (op)
    {
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageFetch:
    case SpvOpImageRead:
    case SpvOpImageWrite:
    case SpvOpImageQuerySizeLod:
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_OP opcode=%u (%s) wc=%u result=%u image=%u",
            op,
            spv_op_name(op),
            wc,
            (wc >= 3) ? ins[2] : 0,
            (wc >= 4) ? ins[3] : 0);
        break;
    default:
        break;
    }
    switch (op)
    {
    /*
     * Type declarations.
     *
     * These must be scanned before variables because
     * later resource classification depends on knowing
     * image/sampled-image relationships.
     */
    case SpvOpTypeFloat:
    case SpvOpTypeInt:
    case SpvOpTypeVector:
    case SpvOpTypeImage:
    case SpvOpTypeSampledImage:
    case SpvOpTypePointer:
        fs_scan_type_instruction(
            s,
            ins,
            op,
            wc);
        STEREO_LOG_VERBOSE(
            "FS_AFTER_TYPE op=%s n_img=%u",
            spv_op_name(op),
            s->n_img);
        STEREO_LOG_VERBOSE(
            "FS_SCAN_STATE_END op=%s n_img=%u",
            spv_op_name(op),
            s->n_img);
        break;
    /*
     * Decorations may appear before OpVariable.
     *
     * Cache them first and apply them when the variable
     * is encountered.
     */
    case SpvOpDecorate:
        fs_process_decoration(
            s,
            ins,
            wc);
        break;
    /*
     * Descriptor/resource declarations.
     */
    case SpvOpVariable:
        fs_scan_variable_instruction(
            s,
            ins,
            wc);
        break;
    /*
     * Function metadata.
     */
    case SpvOpFunction:
        fs_scan_function(
            s,
            ins,
            wc);
        break;
    case SpvOpFunctionParameter:
        fs_scan_function_parameter(
            s,
            ins,
            wc);
        break;
    case SpvOpFunctionEnd:
        s->in_function = false;
        s->current_function_id = 0;
        s->current_param_index = 0;
        STEREO_LOG_VERBOSE(
            "FS_FUNCTION_END");
        break;
    case SpvOpFunctionCall:
        STEREO_LOG_VERBOSE(
            "FS_FUNCTION_CALL wc=%u resultType=%u result=%u function=%u",
            wc,
            wc > 1 ? ins[1] : 0,
            wc > 2 ? ins[2] : 0,
            wc > 3 ? ins[3] : 0);
        fs_scan_function_call(
            s,
            ins,
            wc);
        break;
    /*
     * Resource ownership tracking.
     *
     * Keep SSA ownership for:
     *  - direct loads
     *  - pointer arithmetic / access chains
     *  - simple forwarding ops
     *
     * This is required so a later OpLoad from an access chain can still
     * be traced back to the originating uniform variable.
     */
    case SpvOpAccessChain:
    case SpvOpInBoundsAccessChain:
    case SpvOpPtrAccessChain:
    {
        if (wc >= 4)
        {
            uint32_t result_id = ins[2];
            uint32_t base_id   = ins[3];
            uint32_t owner     = base_id;
            if (!fs_resolve_load_owner(s, base_id, &owner))
            {
                if (fs_var_index(s, base_id) >= 0)
                    owner = base_id;
            }
            STEREO_LOG_VERBOSE(
                "FS_LOAD_RECORD "
                "op=%s "
                "result=%u "
                "owner=%u",
                spv_op_name(op),
                result_id,
                owner);
            fs_add_load_mapping(s, result_id, owner);
            STEREO_LOG_VERBOSE(
                "FS_CHAIN result=%u base=%u owner=%u op=%s",
                result_id,
                base_id,
                owner,
                spv_op_name(op));
        }
        break;
    }
    case SpvOpCopyObject:
    case SpvOpBitcast:
    {
        if (wc >= 4)
        {
            uint32_t result_id = ins[2];
            STEREO_LOG_VERBOSE(
                "FS_COPY_OBJECT "
                "result=%u "
                "src=%u "
                "type=%u",
                result_id,
                ins[3],
                ins[1]);
            uint32_t source_id = ins[3];
            uint32_t owner     = source_id;
            if (!fs_resolve_load_owner(s, source_id, &owner))
            {
                if (fs_var_index(s, source_id) >= 0)
                    owner = source_id;
            }
            STEREO_LOG_VERBOSE(
                "FS_LOAD_RECORD "
                "op=%s "
                "result=%u "
                "owner=%u",
                spv_op_name(op),
                result_id,
                owner);
            fs_add_load_mapping(s, result_id, owner);
            STEREO_LOG_VERBOSE(
                "FS_PROPAGATE_OBJECT op=%s src=%u dst=%u owner=%u",
                spv_op_name(op),
                source_id,
                result_id,
                owner);
        }
        break;
    }
    case SpvOpCompositeExtract:
    {
        if (wc >= 5)
        {
            uint32_t result_id = ins[2];
            uint32_t source_id = ins[3];
            uint32_t owner     = source_id;
            if (!fs_resolve_load_owner(s, source_id, &owner))
            {
                if (fs_var_index(s, source_id) >= 0)
                    owner = source_id;
            }
            STEREO_LOG_VERBOSE(
                "FS_LOAD_RECORD "
                "op=%s "
                "result=%u "
                "owner=%u",
                spv_op_name(op),
                result_id,
                owner);
            fs_add_load_mapping(s, result_id, owner);
        }
        break;
    }
    case SpvOpVectorShuffle:
    {
        if (wc >= 6)
        {
            uint32_t result_id = ins[2];
            uint32_t source_id = ins[3];
            uint32_t owner     = source_id;
            if (!fs_resolve_load_owner(s, source_id, &owner))
            {
                if (fs_var_index(s, source_id) >= 0)
                    owner = source_id;
            }
            STEREO_LOG_VERBOSE(
                "FS_LOAD_RECORD "
                "op=%s "
                "result=%u "
                "owner=%u",
                spv_op_name(op),
                result_id,
                owner);
            fs_add_load_mapping(s, result_id, owner);
        }
        break;
    }
    case SpvOpLoad:
    {
        if (wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_LOAD_SCAN "
                "result=%u "
                "ptr=%u "
                "type=%u",
                ins[2],
                ins[3],
                ins[1]);
        }
        fs_scan_load_instruction(
            s,
            ins,
            wc);
        break;
    }
    case SpvOpSampledImage:
    {
        STEREO_LOG_VERBOSE(
            "FS_ENTER_TRACK_SAMPLED resultType=%u result=%u image=%u sampler=%u",
            (wc >= 2) ? ins[1] : 0,
            (wc >= 3) ? ins[2] : 0,
            (wc >= 4) ? ins[3] : 0,
            (wc >= 5) ? ins[4] : 0);
        STEREO_LOG_VERBOSE(
            "FS_INPUT_OPSAMPLEDIMAGE "
            "result=%u "
            "type=%u "
            "image=%u "
            "sampler=%u",
            (wc >= 3) ? ins[2] : 0,
            (wc >= 2) ? ins[1] : 0,
            (wc >= 4) ? ins[3] : 0,
            (wc >= 5) ? ins[4] : 0);
        if (wc >= 5)
        {
            STEREO_LOG_VERBOSE(
                "FS_SAMPLED_IMAGE_OP resultType=%u result=%u image=%u sampler=%u",
                ins[1],
                ins[2],
                ins[3],
                ins[4]);
            int src = fs_find_load(s, ins[3]);
            STEREO_LOG_VERBOSE(
                "FS_SAMPLED_IMAGE_LOOKUP image=%u load=%d",
                ins[3],
                src);
        }
        fs_track_sampled_image(
            s,
            ins,
            wc);
        break;
    }
    case SpvOpImage:
        fs_track_image_propagation(
            s,
            ins,
            op,
            wc);
        break;
    /*
     * Final image consumers.
     *
     * This is where depth/normal attachment analysis
     * will eventually feed projection correction.
     */
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageFetch:
    case SpvOpImageRead:
    case SpvOpImageWrite:
    case SpvOpImageQuerySizeLod:
        fs_scan_image_operation(
            s,
            ins,
            op,
            wc);
        break;
    default:
        break;
    }
}
/*
 * fs_prescan()
 *
 * High-level SPIR-V prescan dispatcher.
 *
 * The old implementation mixed:
 *
 *   - type discovery
 *   - descriptor tracking
 *   - function analysis
 *   - image ownership propagation
 *   - debug tracing
 *
 * in one large loop.
 *
 * This wrapper keeps the scan order explicit while allowing each
 * analysis stage to remain independently debuggable.
 *
 * Scan order matters:
 *
 *  1. Types must be known before variables can be classified.
 *  2. Decorations may appear before OpVariable, so they are cached.
 *  3. Variables establish descriptor ownership.
 *  4. Function parameters/calls are collected.
 *  5. Loads and image operations propagate ownership.
 *  6. Post-pass fixups resolve deferred relationships.
 */
static void
fs_fixup_function_parameters(
    FsScan *s);
static void
fs_dump_scan_summary(
    const FsScan *s);
static void
fs_prescan(
    FsScan *s,
    const uint32_t *w,
    size_t c)
{
    STEREO_LOG_VERBOSE("FS_PRESCAN_ENTER");
    if (!s || !w || c < 5)
    {
        STEREO_LOG_VERBOSE(
            "FS_PRESCAN_ABORT s=%p w=%p size=%zu",
            s,
            w,
            c);
        return;
    }
    memset(
        s,
        0,
        sizeof(*s));
    STEREO_LOG_VERBOSE(
        "FS_PRESCAN_MODULE ptr=%p bound=%u words=%zu",
        (void *)w,
        w[3],
        c);
    /*
     * SPIR-V module layout:
     *
     *   [0] Magic
     *   [1] Version
     *   [2] Generator
     *   [3] Bound
     *   [4] Schema
     *
     * Instructions begin at word 5.
     */
    for (size_t i = 5; i < c;)
    {
        uint32_t word =
            w[i];
        uint32_t op =
            word & 0xffffu;
        uint32_t wc =
            word >> 16;
        if (wc == 0 ||
            i + wc > c)
        {
            STEREO_LOG_VERBOSE(
                "FS_INVALID_INSTRUCTION offset=%zu wc=%u size=%zu",
                i,
                wc,
                c);
            break;
        }
        fs_scan_instruction(
            s,
            &w[i],
            op,
            wc);
        i += wc;
    }
    STEREO_LOG_VERBOSE(
        "FS_PRESCAN_SCAN_DONE loads=%u calls=%u params=%u",
        s->n_load,
        s->n_call,
        s->n_param);
    /*
     * Resolve deferred parameter ownership.
     */
    fs_fixup_function_parameters(
        s);
    STEREO_LOG_VERBOSE(
        "FS_PARAM_STATE params=%u calls=%u",
        s->n_param,
        s->n_call);
    
    for (uint32_t p = 0; p < s->n_param; ++p)
    {
        STEREO_LOG_VERBOSE(
            "FS_PARAM id=%u index=%u",
            s->params[p].id,
            p);
    }
    for (uint32_t cidx = 0; cidx < s->n_call; ++cidx)
    {
        STEREO_LOG_VERBOSE(
            "FS_CALL param=%u arg=%u",
            s->calls[cidx].parameter_id,
            s->calls[cidx].argument_var);
    }
    STEREO_LOG_VERBOSE(
        "FS_PRESCAN_AFTER_FIXUP loads=%u calls=%u params=%u",
        s->n_load,
        s->n_call,
        s->n_param);
    /*
     * Rewrite loads that still reference function parameter IDs
     * to the caller's descriptor variable.
     */
    for (uint32_t l = 0; l < s->n_load; ++l)
    {
        FsLoadInfo *load =
            &s->loads[l];
        STEREO_LOG_VERBOSE(
            "FS_LOAD_CHECK load=%u owner=%u",
            load->id,
            load->owner_var);
        for (uint32_t cidx = 0;
             cidx < s->n_call;
             ++cidx)
        {
            FsCallInfo *call =
                &s->calls[cidx];
            STEREO_LOG_VERBOSE(
                "FS_CALL_CHECK param=%u arg=%u",
                call->parameter_id,
                call->argument_var);
            if (load->owner_var ==
                call->parameter_id)
            {
                STEREO_LOG_VERBOSE(
                    "FS_LOAD_FINAL_RESOLVE load=%u param=%u owner=%u",
                    load->id,
                    load->owner_var,
                    call->argument_var);
                STEREO_LOG_VERBOSE(
                    "FS_LOAD_RESOLVED load=%u owner=%u",
                    load->id,
                    call->argument_var);
                load->owner_var =
                    call->argument_var;
                break;
            }
        }
    }
    /*
     * Dump the final ownership graph after fixups.
     */
    fs_dump_scan_summary(s);
    FsImageInfo original_images[FS_MAX_IMG];
    uint32_t original_count = s->n_img;
    STEREO_LOG_VERBOSE(
        "FS_IMAGE_REBUILD original=%u",
        original_count);
    memcpy(original_images, s->images,
           original_count * sizeof(FsImageInfo));
    s->n_img = 0;
    for (uint32_t img = 0; img < original_count; ++img)
    {
        const FsImageInfo *src = &original_images[img];
        bool found = false;
        for (uint32_t v = 0; v < s->n_var; ++v)
        {
            if (src->pointer_type &&
                s->vars[v].type == src->pointer_type)
            {
                if (s->n_img >= FS_MAX_IMG)
                    break;
                FsImageInfo *dst = &s->images[s->n_img++];
                *dst = *src;
                dst->replacement_type = 0;
                dst->replacement_pointer_type = 0;
                dst->replacement_sampled_type = 0;
                dst->owner_var = s->vars[v].id;
                dst->binding   = s->vars[v].binding;
                dst->set       = s->vars[v].set;
                dst->stereo =
                    fs_binding_is_stereo_attachment(
                        s,
                        dst->owner_var);
                STEREO_LOG_VERBOSE(
                    "FS_DUP_IMAGE idx=%u image=%u owner=%u binding=%u stereo=%u",
                    s->n_img - 1,
                    dst->id,
                    dst->owner_var,
                    dst->binding,
                    dst->stereo);
                found = true;
            }
        }
        if (!found)
        {
            if (s->n_img >= FS_MAX_IMG)
                break;
            FsImageInfo *dst = &s->images[s->n_img++];
            *dst = *src;
            dst->replacement_type = 0;
            dst->replacement_pointer_type = 0;
            dst->replacement_sampled_type = 0;
            dst->owner_var = UINT32_MAX;
            dst->binding   = UINT32_MAX;
            dst->set       = UINT32_MAX;
        }
    }
    for (uint32_t l = 0; l < s->n_load; ++l)
    {
        const FsLoadInfo *load = &s->loads[l];
        int vi = fs_var_index(s, load->owner_var);
        STEREO_LOG_VERBOSE(
            "FS_FINAL_LOAD load=%u owner=%u set=%u binding=%u storage=%u type=%u",
            load->id,
            load->owner_var,
            (vi >= 0) ? s->vars[vi].set : 0xffffffffu,
            (vi >= 0) ? s->vars[vi].binding : 0xffffffffu,
            (vi >= 0) ? s->vars[vi].storage : 0xffffffffu,
            (vi >= 0) ? s->vars[vi].type : 0xffffffffu);
    }
    for (uint32_t v = 0; v < s->n_var; ++v)
    {
        if (s->vars[v].id == 15)
        {
            STEREO_LOG_VERBOSE(
                "FS_VAR15_FINAL type=%u storage=%u set=%u binding=%u",
                s->vars[v].type,
                s->vars[v].storage,
                s->vars[v].set,
                s->vars[v].binding);
        }
    }
    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_FINAL idx=%u id=%u sampledImage=%u owner=%u binding=%u",
            i,
            s->images[i].id,
            s->images[i].sampled_type,
            s->images[i].owner_var,
            s->images[i].binding);
    }
    STEREO_LOG_VERBOSE(
        "FS_PRESCAN_EXIT");
}
/*
 * fs_prescan_module()
 *
 * Entry point for fragment shader analysis.
 *
 * Responsibilities:
 *
 *   - initialize FsScan state
 *   - perform the SPIR-V instruction scan
 *   - resolve deferred relationships
 *   - emit final diagnostics
 *
 * Keeping this wrapper separate from fs_prescan() allows future
 * multi-pass analysis:
 *
 *   Pass 1:
 *       structural discovery
 *
 *   Pass 2:
 *       descriptor/image provenance
 *
 *   Pass 3:
 *       projection/depth-space classification
 *
 * The future projection-matrix system will use this separation to
 * determine whether a shader samples:
 *
 *   - camera-space data
 *   - screen-space buffers
 *   - depth reconstructed positions
 *   - lighting/deferred intermediates
 */
static bool
fs_prescan_module(
    FsScan *s,
    const uint32_t *w,
    size_t c)
{
    if (!s || !w || c < 5)
    {
        STEREO_LOG_VERBOSE(
            "FS_PRESCAN_INVALID_MODULE");
        return false;
    }
    fs_prescan(
        s,
        w,
        c);
    if (s->n_var == 0 &&
        s->n_img == 0 &&
        s->n_load == 0)
    {
        STEREO_LOG_VERBOSE(
            "FS_PRESCAN_EMPTY_MODULE");
    }
    STEREO_LOG_VERBOSE(
        "FS_PRESCAN_COMPLETE vars=%u images=%u loads=%u functions=%u calls=%u",
        s->n_var,
        s->n_img,
        s->n_load,
        s->n_function,
        s->n_call);
    return true;
}
/*
 * Post-processing
 */
/*
 * Resolve descriptor ownership across function calls.
 *
 * During the initial scan we only know:
 *
 *     caller argument #0  ---> descriptor variable
 *
 * Later, after every function has been scanned, we know:
 *
 *     function parameter ID corresponding to argument #0
 *
 * This pass joins those two pieces of information so image loads
 * performed inside helper functions still resolve back to the
 * original descriptor variable.
 *
 * Before:
 *
 *      load_vars[] --> parameter ID
 *
 * After:
 *
 *      load_vars[] --> descriptor variable
 *
 * This is required because many deferred renderers wrap depth,
 * normal and SSAO sampling inside helper functions.
 */
static void
fs_fixup_function_parameters(
    FsScan *s)
{
    if (!s)
        return;
    /*
     * Resolve parameter ids for every call.
     */
    for (uint32_t i = 0; i < s->n_call; ++i)
    {
        FsCallInfo *call =
            &s->calls[i];
        int fn =
            fs_find_function(
                s,
                call->function_id);
        if (fn < 0)
            continue;
        FsFunctionInfo *func =
            &s->functions[fn];
        uint32_t param_index =
            func->first_param +
            call->parameter_index;
        if (param_index >= s->n_param)
            continue;
        call->parameter_id =
            s->params[param_index].id;
        STEREO_LOG_VERBOSE(
            "FS_CALL_PARAMETER function=%u param=%u arg=%u",
            call->function_id,
            call->parameter_id,
            call->argument_var);
    }
    /*
     * Resolve deferred ownership.
     *
     * owner_var currently contains the parameter SSA id
     * recorded during OpLoad.
     */
    for (uint32_t i = 0; i < s->n_load; ++i)
    {
        FsLoadInfo *load =
            &s->loads[i];
        int p =
            fs_find_parameter(
                s,
                load->owner_var);
        if (p < 0)
            continue;
        FsParameterInfo *param =
            &s->params[p];
        bool resolved = false;
        for (uint32_t c = 0; c < s->n_call; ++c)
        {
            FsCallInfo *call =
                &s->calls[c];
            if (call->function_id !=
                param->function_id)
                continue;
            if (call->parameter_id !=
                param->id)
                continue;
            load->owner_var =
                call->argument_var;
            resolved = true;
            STEREO_LOG_VERBOSE(
                "FS_LOAD_FIXUP load=%u owner=%u",
                load->id,
                load->owner_var);
            break;
        }
        if (!resolved)
        {
            STEREO_LOG_VERBOSE(
                "FS_LOAD_FIXUP_FAILED load=%u param=%u",
                load->id,
                param->id);
        }
    }
    uint32_t unresolved = 0;
    for (uint32_t i = 0; i < s->n_load; ++i)
    {
        int var =
            fs_var_index(
                s,
                s->loads[i].owner_var);
        if (var < 0)
            ++unresolved;
    }
    STEREO_LOG_VERBOSE(
        "FS_FIXUP_COMPLETE loads=%u unresolved=%u",
        s->n_load,
        unresolved);
}
/*
 * Dump the final prescan state.
 *
 * This is called after all instruction scanning and ownership
 * fixups have completed. At this point every descriptor,
 * function parameter and image load should have been resolved
 * to its originating descriptor variable.
 *
 * These logs are invaluable when diagnosing why a sampled image
 * was (or was not) classified as a stereo attachment.
 */
static void
fs_dump_scan_summary(
    const FsScan *s)
{
    if (!s)
        return;
    STEREO_LOG_VERBOSE(
        "========== FS PRESCAN SUMMARY ==========");
    STEREO_LOG_VERBOSE(
        "Images=%u SampledImages=%u Variables=%u Loads=%u Params=%u Functions=%u Calls=%u",
        s->n_img,
        s->n_si,
        s->n_var,
        s->n_load,
        s->n_param,
        s->n_function,
        s->n_call);
    /*
     * Image types
     */
    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        const FsImageInfo *img =
            &s->images[i];
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_FINAL id=%u depth=%u arrayed=%u patchable=%u",
            img->id,
            img->depth,
            img->arrayed,
            img->patchable);
    }
    /*
     * Variables
     */
    for (uint32_t i = 0; i < s->n_var; ++i)
    {
        const FsVariableInfo *v =
            &s->vars[i];
        STEREO_LOG_VERBOSE(
            "FS_VAR_FINAL id=%u type=%u storage=%u set=%u binding=%u location=%u",
            v->id,
            v->type,
            v->storage,
            v->set,
            v->binding,
            v->location);
    }
    /*
     * Decorations
     */
    for (uint32_t i = 0; i < s->n_dec; ++i)
    {
        const FsDecorationInfo *d =
            &s->decorations[i];
        STEREO_LOG_VERBOSE(
            "FS_DEC target=%u set=%u binding=%u",
            d->target,
            d->set,
            d->binding);
    }
    /*
     * Functions
     */
    for (uint32_t i = 0; i < s->n_function; ++i)
    {
        const FsFunctionInfo *fn =
            &s->functions[i];
        STEREO_LOG_VERBOSE(
            "FS_FUNCTION id=%u firstParam=%u",
            fn->id,
            fn->first_param);
    }
    /*
     * Parameters
     */
    for (uint32_t i = 0; i < s->n_param; ++i)
    {
        const FsParameterInfo *p =
            &s->params[i];
        STEREO_LOG_VERBOSE(
            "FS_PARAM id=%u function=%u index=%u",
            p->id,
            p->function_id,
            p->index);
    }
    /*
     * Loads
     */
    for (uint32_t i = 0; i < s->n_load; ++i)
    {
        const FsLoadInfo *load =
            &s->loads[i];
        int var =
            fs_var_index(
                s,
                load->owner_var);
        if (var >= 0)
        {
            STEREO_LOG_VERBOSE(
                "FS_LOAD_FINAL id=%u source=%u owner=%u set=%u binding=%u storage=%u",
                load->id,
                load->source_id,
                load->owner_var,
                s->vars[var].set,
                s->vars[var].binding,
                s->vars[var].storage);
        }
        else
        {
            STEREO_LOG_VERBOSE(
                "FS_LOAD_FINAL id=%u source=%u owner=%u (unresolved)",
                load->id,
                load->source_id,
                load->owner_var);
        }
    }
    /*
     * Calls
     */
    for (uint32_t i = 0; i < s->n_call; ++i)
    {
        const FsCallInfo *call =
            &s->calls[i];
        STEREO_LOG_VERBOSE(
            "FS_CALL_FINAL function=%u parameter=%u argument=%u",
            call->function_id,
            call->parameter_id,
            call->argument_var);
    }
    STEREO_LOG_VERBOSE(
        "========================================");
}

static uint32_t
fs_count_patches(
    const FsScan *s,
    const uint32_t *w,
    size_t c)
{
    uint32_t count = 0;
    bool in_func = false;
    for (size_t i = 5; i < c;)
    {
        uint32_t op = w[i] & 0xffffu;
        uint32_t wc = w[i] >> 16;
        if (!wc || i + wc > c)
            break;
        if (op == SpvOpFunction)
            in_func = true;
        /*
         * Image sampling instructions.
         */
        if (in_func &&
            wc >= 5 &&
            (op == SpvOpImageSampleImplicitLod ||
             op == SpvOpImageSampleExplicitLod ||
             op == SpvOpImageSampleDrefImplicitLod ||
             op == SpvOpImageSampleDrefExplicitLod))
        {
            if (fs_find_load(s, w[i + 3]) >= 0)
            {
                STEREO_LOG_VERBOSE(
                    "FS_PATCH_COUNTER sample image=%u result=%u coord=%u total=%u",
                    w[i + 3],
                    w[i + 2],
                    w[i + 4],
                    count + 1);
                ++count;
            }
        }
        /*
         * ImageFetch
         */
        if (in_func &&
            op == SpvOpImageFetch &&
            wc >= 5)
        {
            uint32_t descriptor_var = 0;
            int load =
                fs_find_load(
                    s,
                    w[i + 3]);
            if (load >= 0)
                descriptor_var =
                    s->loads[load].owner_var;
            if (descriptor_var == 0)
            {
                STEREO_LOG_VERBOSE(
                    "FS_FETCH_NO_DESCRIPTOR image=%u",
                    w[i + 3]);
            }
            STEREO_LOG_VERBOSE(
                "FS_FETCH_CLASSIFY image=%u descriptor=%u",
                w[i + 3],
                descriptor_var);
            if (fs_should_patch_sample(s, hash_spv(w, c), descriptor_var))
            {
                uint32_t binding = 0xffffffffu;
                int var =
                    fs_var_index(
                        s,
                        descriptor_var);
                if (var >= 0)
                    binding =
                        s->vars[var].binding;
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLE_PATCH_APPLY descriptor=%u binding=%u",
                    descriptor_var,
                    binding);
                STEREO_LOG_VERBOSE(
                    "FS_PATCH_COUNTER fetch image=%u result=%u coord=%u total=%u",
                    w[i + 3],
                    w[i + 2],
                    w[i + 4],
                    count + 1);
                ++count;
            }
        }
        i += wc;
    }
    return count;
}

static int
fs_image_index(
    const FsScan *s,
    uint32_t id)
{
    if (!s)
        return -1;

    for (uint32_t i = 0; i < s->n_img; ++i)
    {
        if (s->images[i].id == id)
            return (int)i;
    }

    return -1;
}

static bool
fs_type_is_input_attachment(
    const FsScan *s,
    uint32_t type)
{
    if (!s)
        return false;

    for (uint32_t i = 0; i < s->n_var; ++i)
    {
        if (s->vars[i].type == type &&
            s->vars[i].storage ==
                SpvStorageClassInput)
        {
            return true;
        }
    }

    return false;
}

bool spirv_patch_stereo_fs(
    const uint32_t *in, size_t in_c,
    uint32_t **out, size_t *out_c)
{
    STEREO_LOG_VERBOSE("CALLED spirv_patch_stereo_fs");
    if (!in || in_c < 5 || in[0] != SPIRV_MAGIC) return false;
    STEREO_LOG_VERBOSE(
        "FS_PATCH_ENTER hash=%016llx words=%zu",
        (unsigned long long)hash_spv(in, in_c),
        in_c);
    uint64_t h = hash_spv(in, in_c);
    STEREO_LOG_VERBOSE(
        "FS_PATCH_MODULE hash=%016llx words=%zu",
        (unsigned long long)h,
        in_c);
    FsScan s;
    fs_prescan(&s, in, in_c);
    for (uint32_t ii = 0; ii < s.n_img; ++ii)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_FINAL "
            "idx=%u "
            "image=%u "
            "sampled_type=%u "
            "sampled_type_id=%u "
            "pointer=%u "
            "binding=%u "
            "owner=%u",
            ii,
            s.images[ii].id,
            s.images[ii].sampled_type,
            s.images[ii].sampled_type_id,
            s.images[ii].pointer_type,
            s.images[ii].binding,
            s.images[ii].owner_var);
    }
    for (uint32_t v = 0; v < s.n_var; ++v)
    {
        if (s.vars[v].storage == SpvStorageClassUniformConstant)
        {
            STEREO_LOG_VERBOSE(
                "FS_DESCRIPTOR "
                "id=%u "
                "type=%u "
                "storage=%u "
                "set=%u "
                "binding=%u",
                s.vars[v].id,
                s.vars[v].type,
                s.vars[v].storage,
                s.vars[v].set,
                s.vars[v].binding);
        }
    }
    for (uint32_t i = 0; i < s.n_img; ++i)
    {
        STEREO_LOG_VERBOSE(
            "FS_IMAGE_TABLE "
            "type=%u "
            "sampledType=%u "
            "owner=%u "
            "set=%u "
            "binding=%u",
            s.images[i].id,
            s.images[i].sampled_type,
            s.images[i].owner_var,
            s.images[i].set,
            s.images[i].binding);
    }
    for (uint32_t i = 0; i < s.n_var; ++i)
    {
        const FsVariableInfo *var = &s.vars[i];
        if (var->binding != 0xffffffffu)
        {
            STEREO_LOG_VERBOSE(
                "FS_DESCRIPTOR_SUMMARY var=%u set=%u binding=%u type=%u",
                var->id,
                var->set,
                var->binding,
                var->type);
            /* Dump the descriptor's type chain */
            uint32_t t = var->type;
            while (t)
            {
                uint32_t next = 0;
                for (size_t j = 5; j < in_c;)
                {
                    uint32_t wc = in[j] >> 16;
                    uint32_t op = in[j] & 0xffff;
                    if (!wc || j + wc > in_c)
                        break;
                    if (in[j + 1] == t)
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_TYPE_CHAIN id=%u opcode=%u (%s)",
                            t,
                            op,
                            spv_op_name(op));
                        if (op == SpvOpTypePointer && wc >= 4)
                            next = in[j + 3];
                        else if (op == SpvOpTypeSampledImage && wc >= 3)
                            next = in[j + 2];
                        break;
                    }
                    j += wc;
                }
                t = next;
            }
        }
    }
    if (s.n_img == 0 || !s.float_id)
    {
        STEREO_LOG_VERBOSE(
            "FS_PATCH_REJECT images=%u float_id=%u",
            s.n_img,
            s.float_id);
        return false;
    }
    uint32_t n_patches = fs_count_patches(&s, in, in_c);
    /* Allocate new IDs above current bound */
    uint32_t nid           = in[3];
    uint32_t new_int_id    = s.int_id        ? s.int_id        : nid++;
    STEREO_LOG_VERBOSE("FS_NID_ALLOC assigned=%u next=%u", new_int_id, nid);
    uint32_t new_v3f_id    = s.v3float_id    ? s.v3float_id    : nid++;
    STEREO_LOG_VERBOSE("FS_NID_ALLOC assigned=%u next=%u", new_v3f_id, nid);
    uint32_t new_v3i_id    = s.v3int_id ? s.v3int_id : nid++;
    STEREO_LOG_VERBOSE("FS_NID_ALLOC assigned=%u next=%u", new_v3i_id, nid);
    uint32_t new_v3u_id    = 0;
    if (s.uint_id)
        new_v3u_id = s.v3uint_id ? s.v3uint_id : nid++;
    STEREO_LOG_VERBOSE("FS_NID_ALLOC assigned=%u next=%u", new_v3u_id, nid);
    uint32_t new_pin_id    = s.ptr_int_in_id ? s.ptr_int_in_id : nid++;
    STEREO_LOG_VERBOSE("FS_NID_ALLOC assigned=%u next=%u", new_pin_id, nid);
    uint32_t new_vi_id     = s.vi_var_id     ? s.vi_var_id     : nid++;
    STEREO_LOG_VERBOSE("FS_NID_ALLOC assigned=%u next=%u", new_vi_id, nid);
    uint32_t new_vi_type   = s.int_id ? s.int_id : new_int_id;
    bool     is_new_vi     = (s.vi_var_id == 0);
    bool     emit_vi_decorate  = is_new_vi;
    bool     emit_vi_variable  = is_new_vi;
    STEREO_LOG(
        "FS_VIEW_DECISION is_new_vi=%d vi_var_id=%u new_vi_id=%u emit_decorate=%d emit_variable=%d",
        (int)is_new_vi,
        s.vi_var_id,
        new_vi_id,
        (int)emit_vi_decorate,
        (int)emit_vi_variable);
    STEREO_LOG_VERBOSE(
        "FS_SCAN_SUMMARY int=%u uint=%u v2i=%u v2u=%u v3i=%u v3u=%u",
        s.int_id,
        s.uint_id,
        s.v2int_id,
        s.v2uint_id,
        s.v3int_id,
        s.v3uint_id);
    for (uint32_t img = 0; img < s.n_img; ++img)
    {
        if (!s.images[img].patchable)
            continue;
        uint32_t replacement = 0;
        uint32_t replacement_sampled = 0;
        for (uint32_t prev = 0; prev < img; ++prev)
        {
            if (!s.images[prev].patchable)
                continue;
            if (s.images[prev].sampled_type_id != s.images[img].sampled_type_id)
                continue;
            if (!s.images[prev].replacement_type ||
                !s.images[prev].replacement_sampled_type)
                continue;
            replacement = s.images[prev].replacement_type;
            replacement_sampled = s.images[prev].replacement_sampled_type;
            break;
        }
        if (replacement == 0)
        {
            replacement = nid++;
            replacement_sampled = s.images[img].sampled_type_id;
            STEREO_LOG_VERBOSE(
                "FS_REPLACEMENT_ALLOC "
                "idx=%u "
                "image=%u "
                "sampledType=%u "
                "replacement=%u "
                "replacementSampled=%u",
                img,
                s.images[img].id,
                s.images[img].sampled_type_id,
                replacement,
                replacement_sampled);
        }
        else
        {
            STEREO_LOG_VERBOSE(
                "FS_REPLACEMENT_REUSE "
                "idx=%u "
                "image=%u "
                "sampledType=%u "
                "replacement=%u "
                "replacementSampled=%u",
                img,
                s.images[img].id,
                s.images[img].sampled_type_id,
                replacement,
                replacement_sampled);
        }
        s.images[img].replacement_type = replacement;
        s.images[img].replacement_sampled_type = replacement_sampled;
        STEREO_LOG_VERBOSE(
            "FS_REPLACEMENT_ASSIGN "
            "idx=%u "
            "image=%u "
            "sampledType=%u "
            "owner=%u "
            "binding=%u "
            "replacement=%u "
            "replacementSampled=%u",
            img,
            s.images[img].id,
            s.images[img].sampled_type,
            s.images[img].owner_var,
            s.images[img].binding,
            s.images[img].replacement_type,
            s.images[img].replacement_sampled_type);
        STEREO_LOG_VERBOSE(
            "FS_RESERVE_OWNER image=%u owner=%u binding=%u replacement=%u",
            s.images[img].id,
            s.images[img].owner_var,
            s.images[img].binding,
            s.images[img].replacement_type);
        STEREO_LOG_VERBOSE(
            "FS_RESERVE_ARRAY_TYPE old=%u new=%u owner=%u binding=%u",
            s.images[img].id,
            s.images[img].replacement_type,
            s.images[img].owner_var,
            s.images[img].binding);
        STEREO_LOG_VERBOSE(
            "IMAGE_RESERVED "
            "image=%u "
            "replacementImage=%u "
            "replacementSampled=%u "
            "replacementPointer=%u",
            s.images[img].id,
            s.images[img].replacement_type,
            s.images[img].replacement_sampled_type,
            s.images[img].replacement_pointer_type);
    }
    for (uint32_t img = 0; img < s.n_img; ++img)
    {
        if (!s.images[img].patchable)
            continue;
        s.images[img].replacement_pointer_type = nid++;
    }
    uint32_t samp_nid      = nid;
    uint32_t qsize_nid     = samp_nid + n_patches * 5 + 8;
    /*
     * ImageSample/ImageFetch consume 5 ids.
     * ImageQuerySizeLod consumes only 4 ids,
     * but reserving 5 keeps accounting simple.
     */
    uint32_t new_bound     = samp_nid + n_patches * 5 + 8;
    STEREO_LOG_VERBOSE(
        "FS_NID_INIT bound=%u nid=%u",
        new_bound,
        nid);
    SpvBuf ob;
    if (!sb_init(&ob, in_c + 60 + (size_t)n_patches * 28))
        return false;
    uint32_t id_bound = new_bound;
    bool *emitted_type = calloc(id_bound, sizeof(*emitted_type));
    if (!emitted_type)
    {
        free(emitted_type);
        sb_free(&ob);
        return false;
    }
    bool mv_added   = s.has_mv_cap;
    bool ext_done   = false;
    uint32_t spv_version = in[1];
    bool need_mv_ext =
        !s.has_mv_cap &&
        ((spv_version >> 16) == 1) &&
        (((spv_version >> 8) & 0xff) == 0);
    bool types_done = false;
    bool ep_done    = false;
    bool in_func    = false;
    /* Header */
    sb_push_n(&ob, in, 5);
    ob.w[3] = new_bound;
    for (size_t i = 5; i < in_c; ) {
        uint32_t op = in[i] & 0xffff; uint32_t wc = in[i] >> 16;
        if (!wc || i + wc > in_c) break;
        if (in_func &&
            op >= SpvOpImageSampleImplicitLod &&
            op <= SpvOpImageSparseTexelsResident)
        {
            STEREO_LOG_VERBOSE(
                "FS_IMAGE_OPCODE off=%zu op=%u (%s) wc=%u resultType=%u result=%u sampled=%u coord=%u",
                i,
                op,
                spv_op_name(op),
                wc,
                (wc >= 2) ? in[i + 1] : 0,
                (wc >= 3) ? in[i + 2] : 0,
                (wc >= 4) ? in[i + 3] : 0,
                (wc >= 5) ? in[i + 4] : 0);
            int load = fs_find_load(&s, in[i + 3]);
            STEREO_LOG_VERBOSE(
                "FS_PATCH_BEGIN "
                "sampled=%u "
                "load=%d",
                in[i + 3],
                load);
            STEREO_LOG_VERBOSE(
                "FS_LOAD_LOOKUP sampled=%u load=%d",
                in[i + 3],
                load);
            if (load < 0)
            {
                for (size_t j = 5; j < in_c;)
                {
                    uint32_t word2 = in[j];
                    uint32_t op2 = word2 & 0xffffu;
                    uint32_t wc2 = word2 >> 16;
                    if (wc2 == 0 || j + wc2 > in_c)
                        break;
                    if (wc2 >= 3 && in[j + 2] == in[i + 3])
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_SAMPLE_PRODUCER id=%u op=%u (%s) off=%zu wc=%u",
                            in[i + 3],
                            op2,
                            spv_op_name(op2),
                            j,
                            wc2);
                        if (op2 == SpvOpLoad && wc2 >= 4)
                        {
                            STEREO_LOG_VERBOSE(
                                "FS_PRODUCER_LOAD result=%u type=%u ptr=%u",
                                in[j + 2],
                                in[j + 1],
                                in[j + 3]);
                        }
                        for (uint32_t w = 0; w < wc2; ++w)
                        {
                            STEREO_LOG_VERBOSE(
                                "FS_SAMPLE_PRODUCER_WORD[%u]=%08x",
                                w,
                                in[j + w]);
                        }
                        break;
                    }
                    j += wc2;
                }
            }
        }
        /* Emit MultiView capability immediately before the first non-capability. */
        if (!mv_added &&
            op != SpvOpCapability)
        {
            uint32_t mv[] =
            {
                op_(SpvOpCapability, 2),
                SpvCapabilityMultiView
            };
            sb_push_n(&ob, mv, 2);
            mv_added = true;
            STEREO_LOG(
                "FS_INJ_MULTIVIEW_CAP emitted MultiView capability (FS patcher)");
        }
        /* SPIR-V 1.0 requires SPV_KHR_multiview immediately after capabilities. */
        if (!ext_done &&
            need_mv_ext &&
            op != SpvOpCapability)
        {
            uint32_t e[] =
            {
                op_(SpvOpExtension, 6),
                0x5F565053, /* SPV_ */
                0x5F52484B, /* KHR_ */
                0x746C756D, /* mult */
                0x65697669, /* ivie */
                0x00000077  /* w */
            };
            sb_push_n(&ob, e, 6);
            ext_done = true;
            STEREO_LOG(
                "FS_INJ_MULTIVIEW_EXT emitted SPV_KHR_multiview extension (FS patcher)");
        }
        /*
         * Inject BuiltIn ViewIndex at the beginning of the annotation section,
         * immediately before the first OpDecorate.
         */
        if (emit_vi_decorate &&
            op == SpvOpDecorate)
        {
            uint32_t d[] =
            {
                op_(SpvOpDecorate, 4),
                new_vi_id,
                SpvDecorationBuiltIn,
                SpvBuiltInViewIndex
            };
            sb_push_n(&ob, d, 4);
            STEREO_LOG(
                "FS_INJ_VIEW_DECORATE id=%u builtin=ViewIndex",
                new_vi_id);
            /* only emit once */
            emit_vi_decorate = false;
        }
        if (op == SpvOpDecorate &&
            wc >= 4)
        {
            uint32_t target = in[i + 1];
            uint32_t decoration = in[i + 2];
            if (decoration == SpvDecorationDescriptorSet ||
                decoration == SpvDecorationBinding)
            {
                for (uint32_t img = 0; img < s.n_img; ++img)
                {
                    if (s.images[img].owner_var != target)
                        continue;
                    STEREO_LOG_VERBOSE(
                        "FS_DECORATE_KEEP "
                        "target=%u "
                        "decoration=%u "
                        "value=%u "
                        "binding=%u "
                        "set=%u",
                        target,
                        decoration,
                        wc >= 4 ? in[i + 3] : 0,
                        s.images[img].binding,
                        s.images[img].set);
                    break;
                }
            }
            sb_push_n(&ob, &in[i], wc);
            i += wc;
            continue;
        }
        if (op == SpvOpEntryPoint && !ep_done) {
            ep_done = true;
            if (new_vi_id != s.vi_var_id) {
                sb_push(&ob, ((wc+1)<<16)|SpvOpEntryPoint);
                sb_push_n(&ob, &in[i+1], wc-1);
                sb_push(&ob, new_vi_id);
                STEREO_LOG(
                    "FS_INJ_VIEW_IFACE new_vi=%u appended to entry point interface",
                    new_vi_id);
            } else {
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
            }
            i += wc; continue;
        }
        if (op == SpvOpTypeFunction &&
            wc >= 3)
        {
            uint32_t function_type_id = in[i + 1];
            bool patched = false;
            uint32_t w[64];
            if (wc <= 64)
            {
                memcpy(w, &in[i], wc * sizeof(uint32_t));
                for (uint32_t fn = 0; fn < s.n_function; ++fn)
                {
                    if (s.functions[fn].type_id != function_type_id)
                        continue;
                    uint32_t first_param =
                    s.functions[fn].first_param;
                    for (uint32_t p = 0; p < s.n_param; ++p)
                    {
                        if (s.params[p].function_id !=
                            s.functions[fn].id)
                            continue;
                        if (p < first_param)
                            continue;
                        uint32_t function_param_index =
                        p - first_param;
                        uint32_t operand =
                        3 + function_param_index;
                        if (operand >= wc)
                            break;
                        uint32_t parameter_id =
                        s.params[p].id;
                        uint32_t replacement_pointer = 0;
                        for (uint32_t cidx = 0;
                            cidx < s.n_call;
                            ++cidx)
                        {
                            const FsCallInfo *call =
                            &s.calls[cidx];
                            if (call->parameter_id != parameter_id)
                                continue;
                            for (uint32_t img = 0;
                                img < s.n_img;
                                ++img)
                            {
                                const FsImageInfo *image =
                                &s.images[img];
                                if (image->owner_var !=
                                    call->argument_var)
                                    continue;
                                if (!image->stereo ||
                                    !image->replacement_pointer_type)
                                    continue;
                                replacement_pointer =
                                image->replacement_pointer_type;
                                break;
                            }
                            if (replacement_pointer)
                                break;
                        }
                        if (replacement_pointer &&
                            w[operand] != replacement_pointer)
                        {
                            STEREO_LOG_VERBOSE(
                                "FS_FUNCTION_TYPE_REWRITE "
                                "function=%u "
                                "functionType=%u "
                                "param=%u "
                                "oldType=%u "
                                "newType=%u",
                                s.functions[fn].id,
                                function_type_id,
                                parameter_id,
                                w[operand],
                                replacement_pointer);
                            w[operand] =
                            replacement_pointer;
                            patched = true;
                        }
                    }
                }
                if (patched)
                {
                    sb_push_n(&ob, w, wc);
                    if (w[1] < id_bound)
                        emitted_type[w[1]] = true;
                    i += wc;
                    continue;
                }
            }
        }
        if (op == SpvOpTypeSampledImage &&
            wc >= 3)
        {
            uint32_t sampled_id = in[i + 1];
            uint32_t image_type = in[i + 2];
            uint32_t replacement_image = 0;
            uint32_t replacement_sampled = 0;
            bool patch_sampled = false;
            STEREO_LOG_VERBOSE(
                "FS_SAMPLED_IMAGE_DECL "
                "result=%u "
                "imageType=%u",
                sampled_id,
                image_type);
            STEREO_LOG_VERBOSE(
                "FS_SAMPLED_IMAGE_STATE "
                "result=%u "
                "imageType=%u "
                "emitted=%u",
                sampled_id,
                image_type,
                (sampled_id < id_bound) ? emitted_type[sampled_id] : 0);
            for (uint32_t img = 0; img < s.n_img; ++img)
            {
                if (s.images[img].sampled_type_id != sampled_id)
                    continue;
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLED_IMAGE_MATCH "
                    "idx=%u "
                    "result=%u "
                    "oldImage=%u "
                    "replacementImage=%u "
                    "replacementSampled=%u "
                    "owner=%u "
                    "binding=%u",
                    img,
                    sampled_id,
                    image_type,
                    s.images[img].replacement_type,
                    s.images[img].replacement_sampled_type,
                    s.images[img].owner_var,
                    s.images[img].binding);
                if (!s.images[img].stereo ||
                    !s.images[img].replacement_type)
                    continue;
                replacement_image = s.images[img].replacement_type;
                replacement_sampled =
                    s.images[img].replacement_sampled_type;
                if (replacement_sampled == sampled_id)
                {
                    patch_sampled = true;
                    break;
                }
            }
            if (patch_sampled)
            {
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLED_IMAGE_PATCH "
                    "result=%u "
                    "oldImageType=%u "
                    "newImageType=%u "
                    "replacementSampled=%u",
                    sampled_id,
                    image_type,
                    replacement_image,
                    replacement_sampled);
                uint32_t w[3];
                memcpy(w, &in[i], sizeof(w));
                w[2] = replacement_image;
                sb_push_n(&ob, w, 3);
                if (sampled_id < id_bound)
                    emitted_type[sampled_id] = true;
                i += wc;
                continue;
            }
            if (sampled_id < id_bound && emitted_type[sampled_id])
            {
                i += wc;
                continue;
            }
            sb_push_n(&ob, &in[i], wc);
            if (sampled_id < id_bound)
                emitted_type[sampled_id] = true;
            i += wc;
            continue;
        }
        if (op == SpvOpTypePointer &&
            wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_POINTER_DECL "
                "result=%u "
                "storage=%u "
                "type=%u",
                in[i + 1],
                in[i + 2],
                in[i + 3]);
            bool suppress_original = false;
            for (uint32_t img = 0; img < s.n_img; ++img)
            {
                if (in[i + 3] != s.images[img].sampled_type_id)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_POINTER_SKIP_SAME_TYPE "
                        "idx=%u "
                        "ptrTarget=%u "
                        "replacementSampled=%u "
                        "owner=%u "
                        "binding=%u",
                        img,
                        in[i + 3],
                        s.images[img].replacement_sampled_type,
                        s.images[img].owner_var,
                        s.images[img].binding);
                    continue;
                }
                if (!s.images[img].stereo ||
                    !s.images[img].replacement_sampled_type ||
                    s.images[img].replacement_sampled_type == in[i + 3])
                {
                    continue;
                }
                suppress_original = true;
                break;
            }
            if (!suppress_original)
            {
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
            }
            else
            {
                STEREO_LOG_VERBOSE(
                    "FS_POINTER_SUPPRESS_ORIGINAL "
                    "result=%u "
                    "storage=%u "
                    "oldTarget=%u",
                    in[i + 1],
                    in[i + 2],
                    in[i + 3]);
            }
            for (uint32_t img = 0; img < s.n_img; ++img)
            {
                if (in[i + 3] != s.images[img].sampled_type_id ||
                    !s.images[img].stereo ||
                    !s.images[img].replacement_pointer_type ||
                    !s.images[img].replacement_sampled_type)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_POINTER_SKIP_SAME_POINTER "
                        "idx=%u "
                        "pointer=%u "
                        "owner=%u "
                        "binding=%u",
                        img,
                        in[i + 1],
                        s.images[img].owner_var,
                        s.images[img].binding);
                    continue;
                }
                if (s.images[img].replacement_pointer_type >= id_bound ||
                    s.images[img].replacement_sampled_type >= id_bound)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_POINTER_SKIP_UNDEFINED "
                        "idx=%u "
                        "replacementPointer=%u "
                        "replacementSampled=%u "
                        "idBound=%u "
                        "owner=%u "
                        "binding=%u",
                        img,
                        s.images[img].replacement_pointer_type,
                        s.images[img].replacement_sampled_type,
                        id_bound,
                        s.images[img].owner_var,
                        s.images[img].binding);
                    continue;
                }
                if (!emitted_type[s.images[img].replacement_sampled_type])
                {
                    STEREO_LOG_VERBOSE(
                        "FS_POINTER_SKIP_SAMPLED_UNDEFINED "
                        "idx=%u "
                        "replacementPointer=%u "
                        "replacementSampled=%u "
                        "owner=%u "
                        "binding=%u",
                        img,
                        s.images[img].replacement_pointer_type,
                        s.images[img].replacement_sampled_type,
                        s.images[img].owner_var,
                        s.images[img].binding);
                    continue;
                }
                uint32_t w[4];
                memcpy(w, &in[i], sizeof(w));
                w[1] = s.images[img].replacement_pointer_type;
                w[3] = s.images[img].replacement_sampled_type;
                STEREO_LOG_VERBOSE(
                    "FS_POINTER_EMIT "
                    "idx=%u "
                    "ptrTarget=%u "
                    "replacementSampled=%u "
                    "replacementPointer=%u "
                    "sampledDefined=%u "
                    "owner=%u "
                    "binding=%u",
                    img,
                    in[i + 3],
                    s.images[img].replacement_sampled_type,
                    s.images[img].replacement_pointer_type,
                    s.images[img].replacement_sampled_type < id_bound ?
                    emitted_type[s.images[img].replacement_sampled_type] : 0,
                    s.images[img].owner_var,
                    s.images[img].binding);
                STEREO_LOG_VERBOSE(
                    "FS_POINTER_PATCH "
                    "result=%u "
                    "newResult=%u "
                    "oldType=%u "
                    "newType=%u "
                    "owner=%u "
                    "binding=%u",
                    in[i + 1],
                    w[1],
                    in[i + 3],
                    w[3],
                    s.images[img].owner_var,
                    s.images[img].binding);
                if (emitted_type[w[1]])
                {
                    continue;
                }
                sb_push_n(&ob, w, wc);
                emitted_type[w[1]] = true;
            }
            i += wc;
            continue;
        }
        /* Patch OpTypeImage: Dim=2D Arrayed=0 → Arrayed=1 (in-place word change) */
        if (op == SpvOpTypeImage &&
            wc >= 9 &&
            in[i + 3] == SpvDim2D &&
            in[i + 5] == 0)
        {
            int img_idx = -1;
            for (uint32_t ii = 0; ii < s.n_img; ++ii)
            {
                STEREO_LOG_VERBOSE(
                    "FS_IMAGE_ENTRY idx=%u id=%u owner=%u binding=%u stereo=%u sampledImage=%u",
                    ii,
                    s.images[ii].id,
                    s.images[ii].owner_var,
                    s.images[ii].binding,
                    s.images[ii].stereo,
                    s.images[ii].sampled_type_id);
                if (s.images[ii].id == in[i + 1])
                {
                    img_idx = (int)ii;
                    break;
                }
            }
            if (img_idx >= 0)
            {
                FsImageInfo *img = &s.images[img_idx];
                STEREO_LOG_VERBOSE(
                    "FS_IMAGE_MATCH_BEGIN "
                    "idx=%d "
                    "image=%u "
                    "replacement=%u "
                    "owner=%u "
                    "binding=%u",
                    img_idx,
                    img->id,
                    img->replacement_type,
                    img->owner_var,
                    img->binding);
                uint32_t existing =
                    fs_find_matching_image_type(
                        in,
                        in_c,
                        img->sampled_type,
                        img->dim,
                        img->depth,
                        1,
                        img->ms,
                        img->sampled,
                        img->format);
                STEREO_LOG_VERBOSE(
                    "FS_IMAGE_MATCH_RESULT "
                    "idx=%d "
                    "existing=%u",
                    img_idx,
                    existing);
                STEREO_LOG_VERBOSE(
                    "FS_REUSE_IMAGE_CANDIDATE "
                    "image=%u "
                    "existing=%u "
                    "existingEmitted=%u",
                    img->id,
                    existing,
                    (existing < id_bound) ? emitted_type[existing] : 0);
                if (existing != 0 &&
                    existing < id_bound &&
                    emitted_type[existing])
                {
                    uint32_t existing_sampled =
                    fs_find_matching_sampled_image(
                        in,
                        in_c,
                        existing);
                    if (existing_sampled == 0)
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_REUSE_IMAGE_TYPE_NO_SAMPLED "
                            "image=%u "
                            "existing=%u",
                            img->id,
                            existing);
                    }
                    STEREO_LOG_VERBOSE(
                        "FS_REUSE_IMAGE_TYPE "
                        "image=%u "
                        "existing=%u "
                        "existingSampled=%u "
                        "oldReserved=%u "
                        "oldReservedSampled=%u",
                        img->id,
                        existing,
                        existing_sampled,
                        img->replacement_type,
                        img->replacement_sampled_type);
                    if (existing_sampled == 0 ||
                        existing_sampled >= id_bound ||
                        !emitted_type[existing_sampled])
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_REUSE_IMAGE_REJECT_ORDER "
                            "image=%u "
                            "existing=%u "
                            "existingEmitted=%u "
                            "existingSampled=%u "
                            "sampledEmitted=%u",
                            img->id,
                            existing,
                            (existing < id_bound) ? emitted_type[existing] : 0,
                            existing_sampled,
                            (existing_sampled < id_bound) ?
                            emitted_type[existing_sampled] : 0);
                        sb_push_n(&ob, &in[i], wc);
                        if (in[i + 1] < id_bound)
                        {
                            emitted_type[in[i + 1]] = true;
                        }
                        i += wc;
                        continue;
                    }
                    img->replacement_type = existing;
                    img->replacement_sampled_type = existing_sampled;
                    for (uint32_t copy = 0; copy < s.n_img; ++copy)
                    {
                        if (s.images[copy].sampled_type_id !=
                            img->sampled_type_id)
                            continue;
                        s.images[copy].replacement_type =
                            existing;
                        s.images[copy].replacement_sampled_type =
                            existing_sampled;
                    }
                    STEREO_LOG_VERBOSE(
                        "FS_REUSE_IMAGE_TYPE_FINAL "
                        "image=%u "
                        "replacement=%u "
                        "replacementSampled=%u",
                        img->id,
                        img->replacement_type,
                        img->replacement_sampled_type);
                    /* Keep the original declaration unchanged. */
                    sb_push_n(&ob, &in[i], wc);
                    if (in[i + 1] < id_bound)
                    {
                        emitted_type[in[i + 1]] = true;
                    }
                    i += wc;
                    continue;
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_TYPEIMAGE_RAW "
                "id=%u "
                "sampledType=%u "
                "dim=%u "
                "depth=%u "
                "arrayed=%u "
                "ms=%u "
                "sampled=%u "
                "format=%u",
                in[i + 1],
                in[i + 2],
                in[i + 3],
                in[i + 4],
                in[i + 5],
                in[i + 6],
                in[i + 7],
                in[i + 8]);
            bool patch_this_type = false;
            int patch_img_idx = -1;
            for (uint32_t img = 0; img < s.n_img; ++img)
            {
                if (s.images[img].id != in[i + 1])
                    continue;
                STEREO_LOG_VERBOSE(
                    "FS_IMAGE_TYPE_USER "
                    "type=%u "
                    "owner=%u "
                    "binding=%u "
                    "stereo=%u",
                    s.images[img].id,
                    s.images[img].owner_var,
                    s.images[img].binding,
                    s.images[img].stereo);
                if (s.images[img].stereo)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_PATCH_SELECT "
                        "idx=%u "
                        "image=%u "
                        "sampledType=%u "
                        "replacement=%u",
                        img,
                        s.images[img].id,
                        s.images[img].sampled_type,
                        s.images[img].replacement_type);
                    STEREO_LOG_VERBOSE(
                        "FS_LOAD_WILL_REWRITE "
                        "image=%u "
                        "owner=%u "
                        "binding=%u "
                        "oldType=%u "
                        "newType=%u",
                        s.images[img].id,
                        s.images[img].owner_var,
                        s.images[img].binding,
                        s.images[img].pointer_type,
                        s.images[img].replacement_type);
                    patch_this_type = true;
                    patch_img_idx = (int)img;
                }
            }
            for (uint32_t img = 0; img < s.n_img; ++img)
            {
                if (s.images[img].stereo)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_RESERVED image=%u replacement=%u replacementSampled=%u",
                        s.images[img].id,
                        s.images[img].replacement_type,
                        s.images[img].replacement_sampled_type);
                }
            }
            if (!patch_this_type)
            {
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
                i += wc;
                continue;
            }
            /* Emit the original type unchanged. */
            sb_push_n(&ob, &in[i], wc);
            if (in[i + 1] < id_bound)
            {
                emitted_type[in[i + 1]] = true;
            }
            if (patch_img_idx < 0)
            {
                i += wc;
                continue;
            }
            /* Emit the reserved cloned array type after its component type is defined. */
            uint32_t new_array_type = s.images[patch_img_idx].replacement_type;
            uint32_t new_sampled_type =
                s.images[patch_img_idx].replacement_sampled_type;
            STEREO_LOG_VERBOSE(
                "FS_EMIT_ARRAY "
                "idx=%d "
                "image=%u "
                "replacement=%u "
                "replacementSampled=%u",
                patch_img_idx,
                s.images[patch_img_idx].id,
                new_array_type,
                new_sampled_type);
            STEREO_LOG_VERBOSE(
                "IMAGE_EMIT "
                "oldImage=%u "
                "replacementImage=%u "
                "replacementSampled=%u "
                "replacementPointer=%u "
                "sampledType=%u",
                s.images[patch_img_idx].id,
                s.images[patch_img_idx].replacement_type,
                s.images[patch_img_idx].replacement_sampled_type,
                s.images[patch_img_idx].replacement_pointer_type,
                s.images[patch_img_idx].sampled_type);
            STEREO_LOG_VERBOSE(
                "FS_EMIT_ARRAY_TYPE "
                "image=%u "
                "owner=%u "
                "binding=%u "
                "replacement=%u",
                s.images[patch_img_idx].id,
                s.images[patch_img_idx].owner_var,
                s.images[patch_img_idx].binding,
                new_array_type);
            uint32_t w[9];
            memcpy(w, &in[i], wc * sizeof(uint32_t));
            w[1] = new_array_type;
            w[5] = 1;
            STEREO_LOG_VERBOSE(
                "FS_TYPEIMAGE_PATCH "
                "sampledImageType=%u "
                "oldImageType=%u "
                "newImageType=%u",
                w[1],
                in[i + 2],
                new_array_type);
            STEREO_LOG_VERBOSE(
                "FS_EMIT_ARRAY_IMAGE "
                "result=%u "
                "from=%u",
                w[1],
                in[i + 1]);
            sb_push_n(&ob, w, wc);
            if (w[1] < id_bound)
            {
                emitted_type[w[1]] = true;
            }
            if (new_sampled_type != 0 &&
                new_sampled_type < id_bound &&
                !emitted_type[new_sampled_type])
            {
                uint32_t existing_sampled =
                fs_find_matching_sampled_image(
                    in,
                    in_c,
                    new_array_type);
                bool sampled_is_original = false;
                for (size_t j = 5; j < in_c;)
                {
                    uint32_t wcj = in[j] >> 16;
                    uint32_t opj = in[j] & 0xffff;
                    if (!wcj || j + wcj > in_c)
                        break;
                    if (opj == SpvOpTypeSampledImage &&
                        wcj >= 3 &&
                        in[j + 1] == new_sampled_type)
                    {
                        sampled_is_original = true;
                        break;
                    }
                    j += wcj;
                }
                if (sampled_is_original)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_SKIP_ARRAY_SAMPLED_ORIGINAL "
                        "imageType=%u "
                        "sampledType=%u",
                        new_array_type,
                        new_sampled_type);
                }
                else if (existing_sampled == new_sampled_type)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_RESERVE_ARRAY_SAMPLED "
                        "imageType=%u "
                        "sampledType=%u",
                        new_array_type,
                        new_sampled_type);
                }
                else
                {
                    uint32_t sampled[] =
                    {
                        (3u << 16) | SpvOpTypeSampledImage,
                        new_sampled_type,
                        new_array_type
                    };
                    STEREO_LOG_VERBOSE(
                        "FS_EMIT_ARRAY_SAMPLED "
                        "imageType=%u "
                        "sampledType=%u",
                        new_array_type,
                        new_sampled_type);
                    sb_push_n(&ob, sampled, 3);
                    emitted_type[new_sampled_type] = true;
                }
            }
            i += wc;
            continue;
        }
        if (op == SpvOpFunctionParameter &&
            wc >= 3)
        {
            uint32_t parameter_type = in[i + 1];
            uint32_t parameter_id = in[i + 2];
            uint32_t replacement_pointer = 0;
            uint32_t argument_var = 0;
            for (uint32_t p = 0; p < s.n_param; ++p)
            {
                if (s.params[p].id != parameter_id)
                    continue;
                for (uint32_t cidx = 0; cidx < s.n_call; ++cidx)
                {
                    const FsCallInfo *call = &s.calls[cidx];
                    if (call->parameter_id != parameter_id)
                        continue;
                    argument_var = call->argument_var;
                    for (uint32_t img = 0; img < s.n_img; ++img)
                    {
                        const FsImageInfo *image = &s.images[img];
                        if (image->owner_var != argument_var)
                            continue;
                        if (!image->stereo ||
                            !image->replacement_pointer_type ||
                            !image->replacement_sampled_type)
                            continue;
                        replacement_pointer =
                        image->replacement_pointer_type;
                        STEREO_LOG_VERBOSE(
                            "FS_PARAM_REWRITE "
                            "param=%u "
                            "oldType=%u "
                            "newType=%u "
                            "argument=%u "
                            "owner=%u "
                            "binding=%u",
                            parameter_id,
                            parameter_type,
                            replacement_pointer,
                            argument_var,
                            image->owner_var,
                            image->binding);
                        break;
                    }
                    if (replacement_pointer)
                        break;
                }
                if (replacement_pointer)
                    break;
            }
            if (replacement_pointer)
            {
                uint32_t w[3];
                memcpy(w, &in[i], sizeof(w));
                w[1] = replacement_pointer;
                sb_push_n(&ob, w, wc);
                i += wc;
                continue;
            }
            STEREO_LOG_VERBOSE(
                "FS_PARAM_EMIT_ORIGINAL "
                "param=%u "
                "type=%u",
                parameter_id,
                parameter_type);
            sb_push_n(&ob, &in[i], wc);
            i += wc;
            continue;
        }
        if (op == SpvOpVariable &&
            wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_VAR "
                "id=%u "
                "ptrType=%u "
                "storage=%u",
                in[i + 2],
                in[i + 1],
                in[i + 3]);
            bool patched = false;
            if (in[i + 3] == SpvStorageClassUniformConstant)
            {
                for (uint32_t img = 0; img < s.n_img; ++img)
                {
                    if (s.images[img].owner_var != in[i + 2])
                        continue;
                    if (!s.images[img].replacement_pointer_type ||
                        !s.images[img].replacement_sampled_type)
                        continue;
                    if (s.images[img].replacement_pointer_type >= id_bound ||
                        s.images[img].replacement_sampled_type >= id_bound)
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_VAR_SKIP_UNDEFINED "
                            "var=%u "
                            "replacementPointer=%u "
                            "replacementSampled=%u "
                            "idBound=%u "
                            "set=%u "
                            "binding=%u",
                            in[i + 2],
                            s.images[img].replacement_pointer_type,
                            s.images[img].replacement_sampled_type,
                            id_bound,
                            s.images[img].set,
                            s.images[img].binding);
                        continue;
                    }
                    if (!emitted_type[s.images[img].replacement_pointer_type])
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_VAR_SKIP_POINTER_UNDEFINED "
                            "var=%u "
                            "replacementPointer=%u "
                            "replacementSampled=%u "
                            "set=%u "
                            "binding=%u",
                            in[i + 2],
                            s.images[img].replacement_pointer_type,
                            s.images[img].replacement_sampled_type,
                            s.images[img].set,
                            s.images[img].binding);
                        continue;
                    }
                    uint32_t w[4];
                    memcpy(w, &in[i], sizeof(w));
                    w[1] = s.images[img].replacement_pointer_type;
                    STEREO_LOG_VERBOSE(
                        "FS_VAR_PATCH "
                        "var=%u "
                        "oldPtr=%u "
                        "newPtr=%u "
                        "sampledType=%u "
                        "set=%u "
                        "binding=%u",
                        in[i + 2],
                        in[i + 1],
                        w[1],
                        s.images[img].replacement_sampled_type,
                        s.images[img].set,
                        s.images[img].binding);
                    sb_push_n(&ob, w, wc);
                    patched = true;
                    break;
                }
            }
            if (!patched)
            {
                STEREO_LOG_VERBOSE(
                    "FS_VAR_EMIT_ORIGINAL "
                    "var=%u "
                    "ptrType=%u "
                    "storage=%u",
                    in[i + 2],
                    in[i + 1],
                    in[i + 3]);
                sb_push_n(&ob, &in[i], wc);
            }
            i += wc;
            continue;
        }
        /* Inject new types + gl_ViewIndex variable before first OpFunction */
        if (op == SpvOpFunction && !types_done) {
            types_done = true;
            in_func    = true;
            /* BuiltIn decoration is emitted earlier in the annotation section. */
            if (!s.int_id) {
                uint32_t w[]={(4u<<16)|SpvOpTypeInt, new_int_id, 32, 1};
                sb_push_n(&ob,w,4); }
            if (!s.v3float_id) {
                uint32_t w[]={(4u<<16)|SpvOpTypeVector, new_v3f_id, s.float_id, 3};
                sb_push_n(&ob,w,4); }
            if (!s.v3int_id)
            {
                uint32_t w[]={(4u<<16)|SpvOpTypeVector, new_v3i_id, new_int_id, 3};
                sb_push_n(&ob,w,4);
                s.v3int_id = new_v3i_id;
                STEREO_LOG_VERBOSE(
                    "FS_EMIT_V3INT id=%u scalar=%u existingInt=%u",
                    new_v3i_id,
                    new_int_id,
                    s.int_id);
            }
            if (!s.v3uint_id && s.uint_id)
            {
                STEREO_LOG_VERBOSE(
                    "FS_EMIT_V3UINT id=%u scalar=%u",
                    new_v3u_id,
                    s.uint_id);
                uint32_t w[]={(4u<<16)|SpvOpTypeVector, new_v3u_id, s.uint_id, 3};
                sb_push_n(&ob,w,4);
            }
            STEREO_LOG_VERBOSE(
                "FS_TYPES_FINAL v3i=%u scanV3u=%u newV3u=%u",
                new_v3i_id,
                s.v3uint_id,
                new_v3u_id);
            if (!s.ptr_int_in_id) {
                uint32_t w[]={(4u<<16)|32, new_pin_id, 1, new_int_id};
                sb_push_n(&ob,w,4); }
            if (emit_vi_variable) {
                uint32_t w[]={(4u<<16)|SpvOpVariable, new_pin_id, new_vi_id, SpvStorageClassInput};
                sb_push_n(&ob,w,4);
                STEREO_LOG(
                    "FS_INJ_VIEW_VAR id=%u ptr_type=%u storage=Input",
                    new_vi_id,
                    new_pin_id);
            }
            sb_push_n(&ob, &in[i], wc);
            if (in[i + 1] < id_bound)
            {
                emitted_type[in[i + 1]] = true;
            }
            i += wc; continue;
        }
        if (op == SpvOpFunction) in_func = true;
        if (in_func &&
            op == SpvOpLoad &&
            wc >= 4)
        {
            uint32_t w[4];
            memcpy(w, &in[i], sizeof(w));
            STEREO_LOG_VERBOSE(
                "FS_LOAD_REWRITE_CHECK "
                "ptr=%u",
                in[i + 3]);
            STEREO_LOG_VERBOSE(
                "FS_LOAD "
                "off=%zu "
                "result=%u "
                "type=%u "
                "ptr=%u",
                i,
                w[2],
                w[1],
                w[3]);
            for (uint32_t v = 0; v < s.n_var; ++v)
            {
                if (s.vars[v].id != w[3])
                    continue;
                if (s.vars[v].storage == SpvStorageClassUniformConstant)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_LOAD_MATCH "
                        "ptr=%u "
                        "storage=%u "
                        "binding=%u "
                        "type=%u",
                        s.vars[v].id,
                        s.vars[v].storage,
                        s.vars[v].binding,
                        s.vars[v].type);
                    STEREO_LOG_VERBOSE(
                        "FS_LOAD_PTR "
                        "ptr=%u "
                        "type=%u "
                        "storage=%u "
                        "binding=%u",
                        s.vars[v].id,
                        s.vars[v].type,
                        s.vars[v].storage,
                        s.vars[v].binding);
                    int img = fs_find_image_by_owner(&s, s.vars[v].id);
                    if (img >= 0)
                    {
                        FsImageInfo *image = &s.images[img];
                        STEREO_LOG_VERBOSE(
                            "FS_LOAD_IMAGE "
                            "owner=%u "
                            "binding=%u "
                            "replacementPointer=%u "
                            "replacementSampled=%u",
                            image->owner_var,
                            image->binding,
                            image->replacement_pointer_type,
                            image->replacement_sampled_type);
                        if (w[1] == image->sampled_type_id &&
                            image->replacement_sampled_type)
                        {
                            STEREO_LOG_VERBOSE(
                                "FS_LOAD_PATCH "
                                "result=%u "
                                "oldType=%u "
                                "newType=%u "
                                "binding=%u",
                                w[2],
                                w[1],
                                image->replacement_sampled_type,
                                image->binding);
                            w[1] = image->replacement_sampled_type;
                        }
                        else
                        {
                            STEREO_LOG_VERBOSE(
                                "FS_LOAD_NO_TYPE_PATCH "
                                "result=%u "
                                "type=%u "
                                "sampledType=%u "
                                "replacementSampled=%u "
                                "binding=%u",
                                w[2],
                                w[1],
                                image->sampled_type_id,
                                image->replacement_sampled_type,
                                image->binding);
                        }
                        STEREO_LOG_VERBOSE(
                            "FS_LOAD_KEEP_OWNER "
                            "result=%u "
                            "ptr=%u "
                            "owner=%u "
                            "binding=%u",
                            w[2],
                            w[3],
                            image->owner_var,
                            image->binding);
                    }
                    else
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_LOAD_NO_IMAGE "
                            "ptr=%u",
                            w[3]);
                    }
                    break;
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_LOAD_DECL "
                "resultType=%u "
                "result=%u "
                "pointer=%u",
                in[i + 1],
                in[i + 2],
                in[i + 3]);
            STEREO_LOG_VERBOSE(
                "FS_LOAD_FINAL "
                "result=%u "
                "resultType=%u "
                "ptr=%u",
                w[2],
                w[1],
                w[3]);
            sb_push_n(&ob, w, wc);
            i += wc;
            continue;
        }
        if (op >= SpvOpImageSampleImplicitLod &&
            op <= SpvOpImageSampleProjDrefExplicitLod &&
            wc >= 5)
        {
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_FOUND "
                "off=%zu "
                "opcode=%s "
                "result=%u "
                "sampledImage=%u "
                "coord=%u",
                i,
                spv_op_name(op),
                in[i + 2],
                in[i + 3],
                in[i + 4]);
        }
        if (in_func && op == SpvOpStore && wc >= 3)
        {
            uint32_t target = in[i+1];
            int vi = fs_var_index(&s, target);
            if (vi >= 0)
            {
                STEREO_LOG_VERBOSE(
                    "FS_OUTPUT target=%u set=%u location=%u type=%u value=%u",
                    target,
                    s.vars[vi].set,
                    s.vars[vi].binding,
                    s.vars[vi].type,
                    in[i+2]);
            }
            else
            {
                STEREO_LOG_VERBOSE(
                    "FS_OUTPUT_UNKNOWN target=%u value=%u",
                    target,
                    in[i+2]);
            }
        }
        /* Extend 2D sampling coordinate to 3D for patched loads */
        if (in_func && wc >= 5 &&
            (op == SpvOpImageSampleImplicitLod ||
             op == SpvOpImageSampleExplicitLod ||
             op == SpvOpImageSampleDrefImplicitLod ||
             op == SpvOpImageSampleDrefExplicitLod) &&
            fs_find_load(&s, in[i+3]) >= 0)
        {
            STEREO_LOG_VERBOSE(
                "FS extending sample: op=%u sampledImage=%u coord=%u result=%u",
                op,
                in[i+3],
                in[i+4],
                in[i+2]);
            uint32_t coord_id = in[i+4];
            int coord_type = -1;
            for (uint32_t v = 0; v < s.n_var; ++v)
            {
                if (s.vars[v].id == coord_id)
                {
                    coord_type = s.vars[v].type;
                    break;
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_COORD "
                "coord=%u "
                "type=%d",
                coord_id,
                coord_type);
            uint32_t descriptor_var = 0;
            int load =
                fs_find_load(
                    &s,
                    in[i+3]);
            if (load >= 0)
            {
                descriptor_var =
                    s.loads[load].owner_var;
                int vi =
                    fs_var_index(
                        &s,
                        descriptor_var);
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLE_DESCRIPTOR image=%u descriptorVar=%u set=%u binding=%u",
                    in[i+3],
                    descriptor_var,
                    (vi >= 0) ? s.vars[vi].set : 0xffffffffu,
                    (vi >= 0) ? s.vars[vi].binding : 0xffffffffu);
                if (vi >= 0)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_DESCRIPTOR_TYPE "
                        "descriptor=%u "
                        "type=%u "
                        "storage=%u "
                        "set=%u "
                        "binding=%u",
                        descriptor_var,
                        s.vars[vi].type,
                        s.vars[vi].storage,
                        s.vars[vi].set,
                        s.vars[vi].binding);
                }
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLE_BINDING_DETAIL image=%u descriptor=%u binding=%u",
                    in[i+3],
                    descriptor_var,
                    (vi >= 0) ? s.vars[vi].binding : 0xffffffffu);
                STEREO_LOG_VERBOSE(
                    "FS_SAMPLE_MATCH image=%u load=%d var=%u",
                    in[i+3],
                    load,
                    descriptor_var);
            }
            STEREO_LOG_VERBOSE(
                "FS_SKIP_CANDIDATE "
                "sampledImage=%u "
                "descriptor=%u "
                "result=%u",
                in[i+3],
                descriptor_var,
                in[i+2]);
            if (!fs_should_patch_sample(&s, h, descriptor_var))
            {
                STEREO_LOG_VERBOSE(
                    "FS_PATCH_REJECT "
                    "sampledImage=%u "
                    "descriptorVar=%u "
                    "coord=%u",
                    in[i + 3],
                    descriptor_var,
                    in[i + 4]);
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
                i += wc;
                continue;
            }
            STEREO_LOG_VERBOSE(
                "FS_PATCH_ENTER "
                "sampled=%u "
                "descriptor=%u",
                in[i + 3],
                descriptor_var);
            int vi = fs_var_index(&s, descriptor_var);
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_PATCH_APPLY "
                "hash=%016llx "
                "image=%u "
                "descriptor=%u "
                "set=%u "
                "binding=%u",
                (unsigned long long)h,
                in[i+3],
                descriptor_var,
                (vi >= 0) ? s.vars[vi].set : 0xffffffffu,
                (vi >= 0) ? s.vars[vi].binding : 0xffffffffu);
            int image_type = -1;
            int sampled_type = -1;
            for (uint32_t v = 0; v < s.n_var; ++v)
            {
                if (s.vars[v].id == descriptor_var)
                {
                    sampled_type = s.vars[v].type;
                    break;
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_DESCRIPTOR_TYPES descriptor=%u sampledType=%d",
                descriptor_var,
                sampled_type);
            fs_dump_descriptor_chain(
                &s,
                in,
                in_c,
                descriptor_var);
            STEREO_LOG_VERBOSE(
                "FS_DESCRIPTOR_CHAIN_DONE "
                "descriptor=%u "
                "sampledType=%d",
                descriptor_var,
                sampled_type);
            uint32_t id_lv  = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_lv, samp_nid);
            uint32_t id_cvt = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_cvt, samp_nid);
            uint32_t id_u   = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_u, samp_nid);
            uint32_t id_v   = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_v, samp_nid);
            uint32_t id_c3  = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_c3, samp_nid);
            /* OpLoad %int %vi → id_lv */
            { uint32_t w[]={(4u<<16)|SpvOpLoad, new_int_id, id_lv, new_vi_id};
              sb_push_n(&ob,w,4); }
            /* OpConvertSToF %float id_lv → id_cvt */
            { uint32_t w[]={(4u<<16)|SpvOpConvertSToF, s.float_id, id_cvt, id_lv};
              sb_push_n(&ob,w,4); }
            /* OpCompositeExtract %float coord 0 → id_u */
            { uint32_t w[]={(5u<<16)|SpvOpCompositeExtract, s.float_id, id_u, coord_id, 0};
              sb_push_n(&ob,w,5); }
            /* OpCompositeExtract %float coord 1 → id_v */
            { uint32_t w[]={(5u<<16)|SpvOpCompositeExtract, s.float_id, id_v, coord_id, 1};
              sb_push_n(&ob,w,5); }
            /* OpCompositeConstruct %v3float id_u id_v id_cvt → id_c3 */
            { uint32_t w[]={(6u<<16)|SpvOpCompositeConstruct, new_v3f_id, id_c3, id_u, id_v, id_cvt};
              sb_push_n(&ob,w,6); }
            /* Emit modified sample instruction: word[4] = new coord */
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_REWRITE "
                "result=%u "
                "sampledImage=%u "
                "descriptor=%u "
                "coord2d=%u "
                "coord3d=%u "
                "opcode=%s",
                in[i + 2],
                in[i + 3],
                descriptor_var,
                coord_id,
                id_c3,
                spv_op_name(op));
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_REWRITE_DONE "
                "off=%zu "
                "opcode=%s "
                "result=%u "
                "sampledImage=%u "
                "coord=%u",
                i,
                spv_op_name(op),
                in[i + 2],
                in[i + 3],
                in[i + 4]);
            STEREO_LOG_VERBOSE(
                "FS_COORD_PATCH "
                "off=%zu "
                "oldCoord=%u "
                "newCoord=%u",
                i,
                in[i + 4],
                id_c3);
            STEREO_LOG_VERBOSE(
                "FS_EMIT_SAMPLE "
                "opcode=%s "
                "sampledImage=%u "
                "coordOld=%u "
                "coordNew=%u",
                spv_op_name(op),
                in[i + 3],
                in[i + 4],
                id_c3);
            sb_push(&ob, in[i]);          /* opcode */
            sb_push(&ob, in[i+1]);        /* result type */
            sb_push(&ob, in[i+2]);        /* result id */
            sb_push(&ob, in[i+3]);        /* sampled image (unchanged) */
            sb_push(&ob, id_c3);          /* new 3D coordinate */
            if (wc > 5) sb_push_n(&ob, &in[i+5], wc-5); /* image operands */
            size_t out = ob.n - wc;
            STEREO_LOG_VERBOSE(
                "FS_EMIT_SAMPLE_IDS "
                "sampledImage=%u "
                "coord=%u",
                ob.w[out + 3],
                ob.w[out + 4]);
            STEREO_LOG_VERBOSE(
                "FS_EMIT_WORDS %08x %08x %08x %08x %08x",
                ob.w[out + 0],
                ob.w[out + 1],
                ob.w[out + 2],
                ob.w[out + 3],
                ob.w[out + 4]);
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_WRITTEN "
                "opcode=%s "
                "sampled=%u "
                "coord=%u "
                "wc=%u",
                spv_op_name(op),
                in[i + 3],
                id_c3,
                wc);
            i += wc; continue;
        }
        if (in_func &&
            op == SpvOpImage &&
            wc >= 4)
        {
            if ((in[i + 2] == 170 && in[i + 3] == 169) ||
                (in[i + 2] == 38 && in[i + 3] == 37))
            {
                STEREO_LOG_VERBOSE(
                    "FS_TARGET_IMAGE "
                    "result=%u "
                    "resultType=%u "
                    "sampledImage=%u",
                    in[i + 2],
                    in[i + 1],
                    in[i + 3]);
            }
            STEREO_LOG_VERBOSE(
                "FS_PATCH_IMAGE_VISIT result=%u resultType=%u sampledImage=%u",
                in[i + 2],
                in[i + 1],
                in[i + 3]);
            uint32_t w[4];
            memcpy(w, &in[i], wc * sizeof(uint32_t));
            int load = fs_find_load(&s, in[i + 3]);
            STEREO_LOG_VERBOSE(
                "FS_PATCH_IMAGE_LOADINDEX sampledImage=%u load=%d",
                in[i + 3],
                load);
            if (load < 0)
            {
                for (uint32_t ii = 0; ii < s.n_load; ++ii)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_LOAD_ENTRY "
                        "idx=%u "
                        "id=%u "
                        "owner=%u "
                        "source=%u",
                        ii,
                        s.loads[ii].id,
                        s.loads[ii].owner_var,
                        s.loads[ii].source_id);
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_PATCH_IMAGE load=%d sampledImage=%u",
                load,
                in[i + 3]);
            if (load >= 0)
            {
                STEREO_LOG_VERBOSE(
                    "FS_PATCH_IMAGE_LOAD "
                    "sampledImage=%u "
                    "owner=%u "
                    "binding=%u",
                    in[i + 3],
                    s.loads[load].owner_var,
                    s.loads[load].binding);
                uint32_t owner = s.loads[load].owner_var;
                STEREO_LOG_VERBOSE(
                    "FS_PATCH_OWNER sampledImage=%u owner=%u",
                    in[i + 3],
                    owner);
                for (uint32_t img = 0; img < s.n_img; ++img)
                {
                    STEREO_LOG_VERBOSE(
                        "FS_PATCH_COMPARE "
                        "idx=%u "
                        "imageType=%u "
                        "owner=%u "
                        "stereo=%u "
                        "replacement=%u",
                        img,
                        s.images[img].id,
                        s.images[img].owner_var,
                        s.images[img].stereo,
                        s.images[img].replacement_type);
                    if (s.images[img].owner_var != owner)
                        continue;
                    STEREO_LOG_VERBOSE(
                        "FS_PATCH_OWNER_MATCH idx=%u",
                        img);
                    STEREO_LOG_VERBOSE(
                        "FS_PATCH_TYPES "
                        "idx=%u "
                        "sampledType=%u "
                        "replacementSampled=%u "
                        "replacementImage=%u",
                        img,
                        s.images[img].sampled_type_id,
                        s.images[img].replacement_sampled_type,
                        s.images[img].replacement_type);
                    if (s.images[img].stereo &&
                        s.images[img].replacement_type &&
                        s.images[img].replacement_sampled_type &&
                        s.images[img].replacement_sampled_type)
                    {
                        STEREO_LOG_VERBOSE(
                            "FS_PATCH_IMAGE_REWRITE result=%u oldType=%u newType=%u",
                            w[2],
                            w[1],
                            s.images[img].replacement_type);
                        w[1] = s.images[img].replacement_type;
                        break;
                    }
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_PATCH_IMAGE_EMIT result=%u type=%u sampledImage=%u",
                w[2],
                w[1],
                w[3]);
            sb_push_n(&ob, w, wc);
            if (w[1] < id_bound)
            {
                emitted_type[w[1]] = true;
            }
            i += wc;
            continue;
        }
        if (in_func &&
            (op == SpvOpImageQuerySizeLod || op == SpvOpImageQuerySize) &&
            wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_QUERYSIZE_SCAN image=%u load=%d",
                in[i + 3],
                fs_find_load(&s, in[i + 3]));
        }
        /*
         * OpImageQuerySizeLod
         *
         * Once a 2D image becomes a 2DArray image, ImageQuerySizeLod
         * returns ivec3 instead of ivec2.
         *
         * We keep only xy by inserting a VectorShuffle back to ivec2.
         */
        if (in_func &&
            (op == SpvOpImageQuerySizeLod || op == SpvOpImageQuerySize) &&
            wc >= 4 &&
            fs_find_load(&s, in[i + 3]) >= 0)
        {
            uint32_t descriptor_var = 0;
            uint32_t image_ssa = in[i + 3];
            int load =
                fs_find_load(
                    &s,
                    image_ssa);
            if (load >= 0)
                descriptor_var =
                    s.loads[load].owner_var;
            int img_idx = -1;
            for (uint32_t ii = 0; ii < s.n_img; ++ii)
            {
                if (s.images[ii].owner_var != descriptor_var)
                    continue;
                img_idx = (int)ii;
                STEREO_LOG_VERBOSE(
                    "FS_QSIZE_OWNER_MATCH imageType=%u owner=%u stereo=%u binding=%u",
                    s.images[ii].id,
                    descriptor_var,
                    s.images[ii].stereo,
                    s.images[ii].binding);
                break;
            }
            if (img_idx < 0)
            {
                STEREO_LOG_VERBOSE(
                    "FS_QSIZE_NO_OWNER image=%u descriptor=%u",
                    image_ssa,
                    descriptor_var);
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
                i += wc;
                continue;
            }
            STEREO_LOG_VERBOSE(
                "FS_QSIZE_RESOLVE image=%u load=%d descriptor=%u",
                in[i + 3],
                load,
                descriptor_var);
            if (!s.images[img_idx].stereo)
            {
                STEREO_LOG_VERBOSE(
                    "FS_QSIZE_SKIP image=%u descriptor=%u stereo=%u",
                    in[i + 3],
                    descriptor_var,
                    (img_idx >= 0) ? s.images[img_idx].stereo : 0);
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
                i += wc;
                continue;
            }
            /*
             * The stereo image replacement changes a 2D image into a 2D-array image.
             * OpImageQuerySize* therefore needs a 3-component integer result type:
             *   original 2D image       -> ivec2
             *   stereo 2D-array image   -> ivec3
             *
             * Keep the query instruction itself unchanged apart from its Result Type.
             */
            uint32_t w[5];
            memcpy(w, &in[i], wc * sizeof(uint32_t));
            uint32_t old_result_type = w[1];
            uint32_t old_result_id = w[2];
            uint32_t query_v3_id = qsize_nid++;
            if (!s.v3int_id)
            {
                STEREO_LOG_VERBOSE(
                    "FS_QSIZE_NO_V3INT_TYPE result=%u image=%u",
                    old_result_id,
                    w[3]);
                sb_push_n(&ob, &in[i], wc);
                if (w[1] < id_bound)
                {
                    emitted_type[w[1]] = true;
                }
                i += wc;
                continue;
            }
            w[1] = s.v3int_id;
            w[2] = query_v3_id;
            STEREO_LOG_VERBOSE(
                "FS_REWRITE_QUERYSIZE_V3 "
                "opcode=%s "
                "oldResultType=%u "
                "queryResultType=%u "
                "oldResult=%u "
                "queryResult=%u "
                "qsizeNidNext=%u "
                "image=%u",
                spv_op_name(op),
                old_result_type,
                s.v3int_id,
                old_result_id,
                query_v3_id,
                qsize_nid,
                w[3]);
            sb_push_n(&ob, w, wc);
            uint32_t shuffle[] = {
                (7u << 16) | SpvOpVectorShuffle,
                old_result_type,
                old_result_id,
                query_v3_id,
                query_v3_id,
                0,
                1
            };
            sb_push_n(&ob, shuffle, 7);
            if (old_result_type < id_bound)
            {
                emitted_type[old_result_type] = true;
            }
            if (query_v3_id < id_bound)
            {
                emitted_type[query_v3_id] = true;
            }
            i += wc;
            continue;
        }
        if (in_func && op == SpvOpSampledImage && wc >= 3)
        {
            STEREO_LOG_VERBOSE(
                "FS_IMAGE imageResult=%u sampledImage=%u",
                in[i+2],
                in[i+3]);
        }
        /* Extend OpImageFetch ivec2 -> ivec3(x,y,ViewIndex) */
        if (in_func && op == SpvOpImageFetch && wc >= 5)
        {
            //STEREO_LOG_VERBOSE(
            //    "FS_FETCH opcode image=%u coord=%u result=%u",
            //    in[i+3],
            //    in[i+4],
            //    in[i+2]);
            //STEREO_LOG_VERBOSE(
            //    "FS_FETCH_PATCH_ENTER image=%u result=%u",
            //    in[i+3],
            //    in[i+2]);
            uint32_t coord_id = in[i+4];
            uint32_t descriptor_var = 0;
            bool image_known = false;
            int load =
                fs_find_load(
                    &s,
                    in[i+3]);
            if (load >= 0)
            {
                descriptor_var =
                    s.loads[load].owner_var;
                image_known = true;
                //STEREO_LOG_VERBOSE(
                //    "FS_FETCH_MATCH image=%u loadIndex=%d load=%u var=%u",
                //    in[i+3],
                //    load,
                //    s.loads[load].id,
                //    descriptor_var);
            }
            //STEREO_LOG_VERBOSE(
            //    "FS_FETCH_PATCH_DECISION image=%u known=%u descriptor=%u",
            //    in[i+3],
            //    image_known,
            //    descriptor_var);
            //if (load >= 0)
            //{
            //    STEREO_LOG_VERBOSE(
            //        "FS_FETCH_FOUND image=%u loadIndex=%d var=%u",
            //        in[i+3],
            //        load,
            //        s.loads[load].owner_var);
            //}
            //else
            //{
            //    STEREO_LOG_VERBOSE(
            //        "FS_FETCH_UNKNOWN image=%u",
            //        in[i+3]);
            //}
            int vi = fs_var_index(&s, descriptor_var);
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_PATCH_APPLY hash=%016llx image=%u descriptor=%u set=%u binding=%u",
                (unsigned long long)h,
                in[i+3],
                descriptor_var,
                (vi >= 0) ? s.vars[vi].set : 0xffffffffu,
                (vi >= 0) ? s.vars[vi].binding : 0xffffffffu);
            if (vi >= 0)
            {
                STEREO_LOG_VERBOSE(
                    "FS_FETCH_VAR_INFO image=%u var=%u storage=%u type=%u set=%u binding=%u",
                    in[i+3],
                    descriptor_var,
                    s.vars[vi].storage,
                    s.vars[vi].type,
                    s.vars[vi].set,
                    s.vars[vi].binding);
            }
            STEREO_LOG_VERBOSE(
                "FS_FETCH_DESCRIPTOR image=%u descriptorVar=%u set=%u binding=%u",
                in[i+3],
                descriptor_var,
                (vi >= 0) ? s.vars[vi].set : 0xffffffffu,
                (vi >= 0) ? s.vars[vi].binding : 0xffffffffu);
            if (in[i+3] == 47)
            {
                STEREO_LOG_VERBOSE(
                    "FS_TRACE_IMAGE47 result=%u coord=%u",
                    in[i+2],
                    coord_id);
            }
            if (!fs_binding_is_stereo_attachment(&s, descriptor_var))
            {
                STEREO_LOG_VERBOSE(
                    "FS_FETCH_SKIP_MONO image=%u descriptor=%u binding_not_stereo",
                    in[i+3],
                    descriptor_var);
                sb_push_n(&ob, &in[i], wc);
                if (in[i + 1] < id_bound)
                {
                    emitted_type[in[i + 1]] = true;
                }
                i += wc;
                continue;
            }
            STEREO_LOG_VERBOSE(
                "FS_FETCH_PATCH hash=%016llx image=%u descriptor=%u set=%u binding=%u",
                (unsigned long long)h,
                in[i+3],
                descriptor_var,
                (vi >= 0) ? s.vars[vi].set : 0xffffffffu,
                (vi >= 0) ? s.vars[vi].binding : 0xffffffffu);
            STEREO_LOG_VERBOSE(
                "FS_FETCH_OPCODE opcode=%u image=%u coord=%u result=%u",
                op,
                in[i+3],
                in[i+4],
                in[i+2]);
            STEREO_LOG_VERBOSE(
                "FS_FETCH_STEREO_PATCH image=%u descriptorVar=%u coord=%u",
                in[i+3],
                descriptor_var,
                in[i+4]);
            fs_dump_descriptor_chain(
                &s,
                in,
                in_c,
                descriptor_var);
            uint32_t id_lv = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_lv, samp_nid);
            uint32_t id_x  = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_x, samp_nid);
            uint32_t id_y  = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_y, samp_nid);
            uint32_t id_c3 = samp_nid++;
            STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_c3, samp_nid);
            { uint32_t w[]={(4u<<16)|SpvOpLoad, new_vi_type, id_lv, new_vi_id};
              sb_push_n(&ob,w,4); }
            STEREO_LOG_VERBOSE(
                "FS_VIEWINDEX_LOAD result=%u actualType=%u finalType=%u",
                id_lv,
                new_vi_type,
                new_int_id);
            uint32_t id_layer = id_lv;
            uint32_t coord_scalar_type = new_int_id;
            uint32_t coord_vector_type = new_v3i_id;
            uint32_t coord_type =
                fs_result_type_of(&s, in, in_c, coord_id);
            bool coord_is_uint =
                coord_type &&
                s.uint_id &&
                (coord_type == s.v2uint_id ||
                 coord_type == s.v3uint_id);
            if (coord_is_uint)
            {
                if (!new_v3u_id)
                    break;
                coord_scalar_type = s.uint_id;
                coord_vector_type = new_v3u_id;
                if (new_vi_type != s.uint_id)
                {
                    id_layer = samp_nid++;
                    STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_layer, samp_nid);
                    uint32_t w[] = {
                        (4u << 16) | SpvOpBitcast,
                        s.uint_id,
                        id_layer,
                        id_lv
                    };
                    sb_push_n(&ob, w, 4);
                }
            }
            else if (new_vi_type != new_int_id)
            {
                id_layer = samp_nid++;
                STEREO_LOG_VERBOSE("FS_SAMPNID_ALLOC assigned=%u next=%u", id_layer, samp_nid);
                uint32_t w[] = {
                    (4u << 16) | SpvOpBitcast,
                    new_int_id,
                    id_layer,
                    id_lv
                };
                sb_push_n(&ob, w, 4);
            }
            STEREO_LOG_VERBOSE(
                "FS_COORD_CONSTRUCT "
                "coord=%u "
                "coordType=%u "
                "scalar=%u "
                "vector=%u "
                "viewLoadType=%u "
                "layer=%u "
                "x=%u "
                "y=%u",
                coord_id,
                coord_type,
                coord_scalar_type,
                coord_vector_type,
                new_vi_type,
                id_layer,
                id_x,
                id_y);
            { uint32_t w[]={(5u<<16)|SpvOpCompositeExtract, coord_scalar_type, id_x, coord_id, 0};
              sb_push_n(&ob,w,5); }
            { uint32_t w[]={(5u<<16)|SpvOpCompositeExtract, coord_scalar_type, id_y, coord_id, 1};
              sb_push_n(&ob,w,5); }
            { uint32_t w[]={(6u<<16)|SpvOpCompositeConstruct, coord_vector_type, id_c3,
                            id_x, id_y, id_layer};
              sb_push_n(&ob,w,6); }
            sb_push(&ob, in[i]);
            sb_push(&ob, in[i+1]);
            sb_push(&ob, in[i+2]);
            sb_push(&ob, in[i+3]);
            sb_push(&ob, id_c3);
            if (wc > 5)
                sb_push_n(&ob, &in[i+5], wc - 5);
            STEREO_LOG_VERBOSE(
                "FS_FETCH_PATCHED "
                "result=%u "
                "image=%u "
                "coordOld=%u "
                "coordNew=%u",
                in[i + 2],
                in[i + 3],
                coord_id,
                id_c3);
            STEREO_LOG_VERBOSE(
                "FS_FETCH_PATCH_DONE image=%u newCoord=%u",
                in[i+3],
                id_c3);
            i += wc;
            continue;
        }
        if (op >= SpvOpImageSampleImplicitLod &&
            op <= SpvOpImageSampleProjDrefExplicitLod)
        {
            STEREO_LOG_VERBOSE(
                "FS_SAMPLE_NOT_PATCHED "
                "opcode=%s "
                "resultType=%u "
                "result=%u "
                "sampledImage=%u "
                "coord=%u",
                spv_op_name(op),
                (wc >= 2) ? in[i + 1] : 0,
                (wc >= 3) ? in[i + 2] : 0,
                (wc >= 4) ? in[i + 3] : 0,
                (wc >= 5) ? in[i + 4] : 0);
        }
        sb_push_n(&ob, &in[i], wc);
        if (in[i + 1] < id_bound)
        {
            emitted_type[in[i + 1]] = true;
        }
        i += wc;
    }
    for (size_t j = 5; j < ob.n; )
    {
        uint32_t wc = ob.w[j] >> 16;
        uint32_t op = ob.w[j] & 0xffff;
        if (!wc || j + wc > ob.n)
            break;
        if (op == SpvOpTypeImage && wc >= 9)
        {
            STEREO_LOG_VERBOSE(
                "FS_OUT_TYPEIMAGE "
                "result=%u "
                "sampledType=%u "
                "dim=%u "
                "arrayed=%u",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3],
                ob.w[j + 5]);
            STEREO_LOG_VERBOSE(
                "FS_OUTPUT_IMAGE_TYPE id=%u sampledType=%u dim=%u depth=%u arrayed=%u ms=%u sampled=%u format=%u",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3],
                ob.w[j + 4],
                ob.w[j + 5],
                ob.w[j + 6],
                ob.w[j + 7],
                ob.w[j + 8]);
        }
        if (op == SpvOpImageFetch && wc >= 5)
        {
            STEREO_LOG_VERBOSE(
                "FS_OUT_FETCH "
                "resultType=%u "
                "result=%u "
                "image=%u "
                "coord=%u",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3],
                ob.w[j + 4]);
        }
        if (op == SpvOpTypeSampledImage && wc >= 3)
        {
            STEREO_LOG_VERBOSE(
                "FS_OUT_TYPESAMPLED "
                "result=%u "
                "imageType=%u",
                ob.w[j + 1],
                ob.w[j + 2]);
        }
        if (op == SpvOpSampledImage && wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_OUTPUT_OPSAMPLEDIMAGE "
                "result=%u "
                "type=%u "
                "image=%u "
                "sampler=%u",
                ob.w[j + 2],
                ob.w[j + 1],
                ob.w[j + 3],
                (wc >= 5) ? ob.w[j + 4] : 0);
            STEREO_LOG_VERBOSE(
                "FS_OUT_SAMPLEDIMAGE "
                "result=%u "
                "type=%u "
                "imageType=%u",
                ob.w[j + 2],
                ob.w[j + 1],
                ob.w[j + 3]);
        }
        if ((op == SpvOpCopyObject || op == SpvOpBitcast) && wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_PRODUCER_COPY "
                "result=%u "
                "src=%u "
                "type=%u",
                ob.w[j + 2],
                ob.w[j + 3],
                ob.w[j + 1]);
        }
        if (op == SpvOpLoad && wc >= 4)
        {
            uint32_t ptr_type = 0;
            for (uint32_t v = 0; v < s.n_var; ++v)
            {
                if (s.vars[v].id == ob.w[j + 3])
                {
                    ptr_type = s.vars[v].type;
                    break;
                }
            }
            STEREO_LOG_VERBOSE(
                "FS_OUT_LOAD "
                "result=%u "
                "resultType=%u "
                "ptr=%u "
                "ptrType=%u",
                ob.w[j + 2],
                ob.w[j + 1],
                ob.w[j + 3],
                ptr_type);
        }
        if (op == SpvOpImage && wc >= 4)
        {
            STEREO_LOG_VERBOSE(
                "FS_OUT_IMAGE "
                "result=%u "
                "type=%u "
                "sampledImage=%u",
                ob.w[j + 2],
                ob.w[j + 1],
                ob.w[j + 3]);
        }
        if (op >= SpvOpImageSampleImplicitLod &&
            op <= SpvOpImageSampleProjDrefExplicitLod &&
            wc >= 5)
        {
            STEREO_LOG_VERBOSE(
                "FS_OUT_SAMPLE "
                "resultType=%u "
                "result=%u "
                "sampledImage=%u "
                "coord=%u "
                "opcode=%s",
                ob.w[j + 1],
                ob.w[j + 2],
                ob.w[j + 3],
                ob.w[j + 4],
                spv_op_name(op));
            STEREO_LOG_VERBOSE(
                "FS_PATCHED_SAMPLE "
                "off=%zu "
                "opcode=%s "
                "result=%u "
                "sampledImage=%u "
                "coord=%u "
                "resultType=%u",
                j,
                spv_op_name(op),
                ob.w[j + 2],
                ob.w[j + 3],
                ob.w[j + 4],
                ob.w[j + 1]);
        }
        if (wc >= 3 &&
            (ob.w[j + 2] == 14 ||
             ob.w[j + 2] == 24 ||
             ob.w[j + 2] == 37 ||
             ob.w[j + 2] == 58 ||
             ob.w[j + 2] == 66 ||
             ob.w[j + 2] == 71 ||
             ob.w[j + 2] == 174))
        {
            STEREO_LOG_VERBOSE(
                "FS_PRODUCER "
                "result=%u "
                "opcode=%s "
                "type=%u "
                "wc=%u",
                ob.w[j + 2],
                spv_op_name(op),
                ob.w[j + 1],
                wc);
        }
        j += wc;
    }
    if (nid > samp_nid)
        samp_nid = nid;
    ob.w[3] = qsize_nid;
    *out   = ob.w;
    *out_c = ob.n;
    STEREO_LOG_VERBOSE("FS patched: %u 2D img types→arr, %u samples extended, bound %u→%u",
               s.n_img, n_patches, in[3], ob.w[3]);
    STEREO_LOG_VERBOSE(
        "FS_PATCH_DONE hash=%016llx words=%zu new_words=%zu",
        (unsigned long long)hash_spv(in, in_c),
        in_c,
        ob.n);
    STEREO_LOG_VERBOSE(
        "FS_FINAL_BOUND old=%u new=%u nid=%u samp_nid=%u qsize_nid=%u",
        in[3],
        ob.w[3],
        nid,
        samp_nid,
        qsize_nid);
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

static StereoShaderCache *
cache_find(StereoDevice *sd, VkShaderModule mod)
{
    if (!sd)
        return NULL;
    for (uint32_t i = 0; i < sd->shader_cache_count; i++)
    {
        StereoShaderCache *e = &sd->shader_cache[i];
        if (e->handle == mod)
        {
            STEREO_LOG_VERBOSE(
                "CACHE_FIND_HIT module=%p hash=%016llx words=%zu",
                (void *)mod,
                (unsigned long long)hash_spv(e->spv, e->words),
                e->words);
            return e;
        }
    }
    STEREO_LOG_VERBOSE(
        "CACHE_FIND_MISS module=%p",
        (void *)mod);
    return NULL;
}

static void cache_add(StereoDevice *sd, VkShaderModule h,
                      const uint32_t *spv, size_t words) {
    if (sd->shader_cache_count>=MAX_SHADER_CACHE) return;
    uint32_t *cp=malloc(words*4); if (!cp) return;
    memcpy(cp,spv,words*4);
    CHECK_ARRAY_COUNT(sd->shader_cache_count, MAX_SHADER_CACHE, "shader_cache_count");
    StereoShaderCache *e=&sd->shader_cache[sd->shader_cache_count++];
    e->handle=h; e->spv=cp; e->words=words;
}
static void cache_remove(StereoDevice *sd, VkShaderModule h)
{
    for (uint32_t i = 0; i < sd->shader_cache_count; i++)
    {
        if (sd->shader_cache[i].handle == h)
        {
            free(sd->shader_cache[i].spv);
            uint32_t last = --sd->shader_cache_count;
            if (i != last)
                sd->shader_cache[i] = sd->shader_cache[last];
            memset(&sd->shader_cache[last], 0,
                   sizeof(sd->shader_cache[last]));
            return;
        }
    }
}

/* ── vkCreateShaderModule ─────────────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCI,
                          const VkAllocationCallbacks *pAlloc, VkShaderModule *pSM)
{
    STEREO_LOG_ONCE("FIRST_CALL stereo_CreateShaderModule codeSize=%zu", pCI->codeSize);
    STEREO_LOG_VERBOSE("CALLED stereo_CreateShaderModule");
    StereoDevice *sd=stereo_device_from_handle(device);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    STEREO_LOG_VERBOSE("CALL real CreateShaderModule");
    VkResult res=sd->real.CreateShaderModule(sd->real_device,pCI,pAlloc,pSM);
    STEREO_LOG_VERBOSE("RETURN real CreateShaderModule result=%d", res);
    if (res!=VK_SUCCESS) return res;
    if (!sd->stereo.enabled) return VK_SUCCESS;
    const uint32_t *spv = (const uint32_t *)pCI->pCode;
    size_t wc = pCI->codeSize / 4;
    uint64_t h = hash_spv(spv, wc);
    STEREO_LOG_VERBOSE(
        "CREATE_SHADER module=%p hash=%016llx words=%zu patchable=%u",
        (void *)*pSM,
        (unsigned long long)h,
        wc,
        is_patchable_spv(spv, wc));
    STEREO_LOG_VERBOSE(
        "SHADER_MODULE words=%u hash=%016llx",
        wc,
        (unsigned long long)h);
    const char *dump = stereo_getenv("VKS3D_DUMP_SPIRV");
    if (dump)
    {
        char dp[512];
        _snprintf(
            dp,
            sizeof(dp) - 1,
            "%s\\%016llx.spv",
            dump,
            (unsigned long long)h);
        FILE *f = fopen(dp, "rb");
        if (!f)
        {
            f = fopen(dp, "wb");
            if (f)
            {
                fwrite(
                    spv,
                    4,
                    wc,
                    f);
                fclose(f);
            }
        }
        else
        {
            fclose(f);
        }
    }
    if (is_patchable_spv(spv, wc))
    {
        cache_add(sd, *pSM, spv, wc);
    }
    STEREO_LOG_VERBOSE(
        "CREATE_SHADER_DONE module=%p hash=%016llx",
        (void *)*pSM,
        (unsigned long long)h);
    return VK_SUCCESS;
}

#ifndef VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
#define VK_DYNAMIC_STATE_VERTEX_INPUT_EXT 1000352003
#endif
#ifndef VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT
#define VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT 1000352004
#endif

/* Returns true when the pipeline declares VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
 * (or its related stride dynamic state).  Yuzu and other Switch emulators
 * frequently use dynamic vertex input (vertex pulling via SSBO), which
 * means pVertexInputState is NULL even though the pipeline renders actual
 * world geometry (not a fullscreen quad). */
static bool
pipe_has_dynamic_vtx_input(const VkGraphicsPipelineCreateInfo *ci)
{
    if (!ci || !ci->pDynamicState) return false;
    const VkPipelineDynamicStateCreateInfo *d = ci->pDynamicState;
    if (!d->pDynamicStates) return false;
    for (uint32_t i = 0; i < d->dynamicStateCount; ++i)
    {
        VkDynamicState s = d->pDynamicStates[i];
        if (s == VK_DYNAMIC_STATE_VERTEX_INPUT_EXT ||
            s == VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT ||
            /* Draft/pre-release numbers used by early adopters. */
            s == 93 || s == 94)
        {
            return true;
        }
    }
    return false;
}

static bool
pipe_is_quad_pipeline(const VkGraphicsPipelineCreateInfo *ci)
{
    if (pipe_has_dynamic_vtx_input(ci)) return false;
    if (!ci->pVertexInputState) return true;
    return (ci->pVertexInputState->vertexBindingDescriptionCount == 0);
}

static bool
pipe_is_points_pipeline(const VkGraphicsPipelineCreateInfo *ci)
{
    /* If pInputAssemblyState is missing or topology is dynamic, assume not
     * points (most common case is triangles).  Point-list topologies are
     * rare in main scene rendering so this heuristic is very safe. */
    if (!ci->pInputAssemblyState) return false;
    VkPrimitiveTopology t = ci->pInputAssemblyState->topology;
    if (t == VK_PRIMITIVE_TOPOLOGY_POINT_LIST) return true;
    return false;
}

static uint32_t
pipe_vtx_binding_count(const VkGraphicsPipelineCreateInfo *ci)
{
    if (pipe_has_dynamic_vtx_input(ci)) return 1; /* non-zero -> non-quad */
    if (!ci->pVertexInputState) return 0;
    return ci->pVertexInputState->vertexBindingDescriptionCount;
}

/* ── Debug FS: solid red (ViewIndex=0) / green (ViewIndex=1) ──────────────
 * When VKS3D_DEBUG_VIEW=1, multiview pipelines get their FS replaced by
 * this minimal shader so the user can instantly see whether gl_ViewIndex
 * actually alternates between 0 and 1 per layer.
 * Left half red  = ViewIndex=0 works
 * Right half green = ViewIndex=1 works
 * Both same colour = ViewIndex broken (driver or pipeline issue)
 */
static bool
build_debug_fs_spv(uint32_t **out_spv, size_t *out_wc)
{
    /* IDs: 1=void 2=funcType 3=float 4=int32 5=bool 6=vec4
     *      7=ptrOut 8=ptrIn 9=outColor 10=viewIdx
     *      11=f0 12=f1 13=i0 14=red 15=green
     *      16=main 17=label 18=vi 19=isLeft 20=color  Bound=21 */
    static const uint32_t spv[] = {
        0x07230203, 0x00010600, 0x00000000, 21, 0,           /* header (SPIR-V 1.6) */
        (2u<<16)|17u, 1,                                        /* OpCapability Shader */
        (2u<<16)|17u, 4439,                                    /* OpCapability MultiView */
        (3u<<16)|14u, 0, 1,                                    /* OpMemoryModel Logical GLSL450 */
        (7u<<16)|15u, 4, 16, 0x6E69616D, 0, 9, 10,           /* OpEntryPoint Fragment %16 "main" %9 %10 */
        (3u<<16)|16u, 16, 7,                                   /* OpExecutionMode OriginUpperLeft */
        (4u<<16)|71u, 9, 30, 0,                                /* OpDecorate %9 Location 0 */
        (4u<<16)|71u, 10, 11, 4440,                            /* OpDecorate %10 BuiltIn ViewIndex */
        (2u<<16)|19u, 1,                                       /* OpTypeVoid */
        (3u<<16)|33u, 2, 1,                                    /* OpTypeFunction */
        (3u<<16)|22u, 3, 32,                                   /* OpTypeFloat 32 */
        (4u<<16)|21u, 4, 32, 0,                                /* OpTypeInt 32 unsigned */
        (2u<<16)|20u, 5,                                       /* OpTypeBool */
        (4u<<16)|23u, 6, 3, 4,                                 /* OpTypeVector vec4 */
        (4u<<16)|32u, 7, 3, 6,                                 /* OpTypePointer Output vec4 */
        (4u<<16)|32u, 8, 1, 4,                                 /* OpTypePointer Input int */
        (4u<<16)|43u, 3, 11, 0x00000000,                       /* OpConstant float 0.0 */
        (4u<<16)|43u, 3, 12, 0x3F800000,                       /* OpConstant float 1.0 */
        (4u<<16)|43u, 4, 13, 0,                                /* OpConstant int 0 */
        (7u<<16)|44u, 6, 14, 12, 11, 11, 12,                   /* OpConstantComposite red=(1,0,0,1) */
        (7u<<16)|44u, 6, 15, 11, 12, 11, 12,                   /* OpConstantComposite green=(0,1,0,1) */
        (4u<<16)|59u, 7, 9, 3,                                 /* OpVariable Output */
        (4u<<16)|59u, 8, 10, 1,                                /* OpVariable Input */
        (5u<<16)|54u, 1, 16, 0, 2,                             /* OpFunction */
        (2u<<16)|248u, 17,                                     /* OpLabel */
        (4u<<16)|61u, 4, 18, 10,                               /* OpLoad view_idx */
        (5u<<16)|170u, 5, 19, 18, 13,                          /* OpIEqual %19 = (%18 == 0) */
        (6u<<16)|169u, 6, 20, 19, 14, 15,                     /* OpSelect %20 = %19 ? red : green */
        (3u<<16)|62u, 9, 20,                                   /* OpStore out_color */
        (1u<<16)|253u,                                         /* OpReturn */
        (1u<<16)|56u,                                          /* OpFunctionEnd */
    };
    *out_wc = sizeof(spv) / sizeof(spv[0]);
    *out_spv = (uint32_t *)malloc(*out_wc * 4);
    if (!*out_spv) return false;
    memcpy(*out_spv, spv, *out_wc * 4);
    return true;
}

static bool s_debug_view_checked = false;
static bool s_debug_view_enabled = false;

static bool is_debug_view_enabled(void)
{
    if (!s_debug_view_checked)
    {
        const char *e = getenv("VKS3D_DEBUG_VIEW");
        s_debug_view_enabled = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T'));
        s_debug_view_checked = true;
        if (s_debug_view_enabled)
            STEREO_LOG("DEBUG_VIEW enabled — FS replaced with red/green ViewIndex test");
    }
    return s_debug_view_enabled;
}

/* ── vkCreateGraphicsPipelines ───────────────────────────────────────────── */
VKAPI_ATTR VkResult VKAPI_CALL
stereo_CreateGraphicsPipelines(VkDevice device, VkPipelineCache pc,
    uint32_t N, const VkGraphicsPipelineCreateInfo *pCI,
    const VkAllocationCallbacks *pAlloc, VkPipeline *pP)
{
    STEREO_LOG_VERBOSE(
        "CALLED stereo_CreateGraphicsPipelines this=%p",
        (void*)&stereo_CreateGraphicsPipelines);
    StereoDevice *sd=stereo_device_from_handle(device);
    if (!sd) return VK_ERROR_DEVICE_LOST;
    STEREO_LOG_VERBOSE("PIPE_IN_RAW N=%u pCI=%p first=%p renderPass=%p stageCount=%u pNext=%p",
               N,
               (void*)pCI,
               (N > 0 ? (void*)pCI[0].renderPass : NULL),
               (N > 0 ? (void*)pCI[0].renderPass : NULL),
               (N > 0 ? pCI[0].stageCount : 0),
               (N > 0 ? pCI[0].pNext : NULL));
    STEREO_LOG(
        "CREATE_GFX_PIPELINES count=%u rp=%p stages=%u pNext=%p enabled=%d",
        N,
        (N > 0 ? (void*)pCI[0].renderPass : NULL),
        (N > 0 ? pCI[0].stageCount : 0),
        (N > 0 ? pCI[0].pNext : NULL),
        (int)sd->stereo.enabled);
    if (!sd->stereo.enabled)
        return sd->real.CreateGraphicsPipelines(sd->real_device,pc,N,pCI,pAlloc,pP);
    VkShaderModule                   *tmp_mod     = calloc(N, sizeof(VkShaderModule));
    VkShaderModule                   *tmp_mod_tcs = calloc(N, sizeof(VkShaderModule));
    VkPipelineShaderStageCreateInfo **tst         = calloc(N, sizeof(void*));
    VkGraphicsPipelineCreateInfo     *infos       = malloc(N * sizeof(*infos));
    VkPipelineTessellationStateCreateInfo *tess_infos = calloc(N, sizeof(*tess_infos));
    StereoDebugCtx                   *dbg_out     = calloc(N, sizeof(*dbg_out));
    for (uint32_t i = 0; i < N; i++)
    {
        dbg_out[i].proj_set             = UINT32_MAX;
        dbg_out[i].proj_binding         = UINT32_MAX;
        dbg_out[i].proj_member_mask     = UINT32_MAX;
        dbg_out[i].proj_var             = UINT32_MAX;
    }
    if (!tmp_mod||!tmp_mod_tcs||!tst||!infos||!tess_infos||!dbg_out) {
        free(tmp_mod); free(tmp_mod_tcs); free(tst); free(infos); free(tess_infos); free(dbg_out);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    memcpy(infos, pCI, N * sizeof(*infos));
    for (uint32_t p = 0; p < N; p++) {
        STEREO_LOG_VERBOSE(
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
        "[PATCH] lo=%f ro=%f conv=%f flip=%d sep=%f",
        lo,
        ro,
        conv,
        sd->stereo.flip_eyes,
        sd->stereo.separation);
    for (uint32_t p=0; p<N; p++) {
        const VkGraphicsPipelineCreateInfo *ci=&pCI[p];
        //REMOVED StereoPipelineInfo *info =
        //REMOVED     add_pipeline_info(sd);
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
                 STEREO_LOG_VERBOSE(
                  "PIPE_RENDERING_UPGRADE p=%u viewMask 0x0->0x3 colors=%u depth=%u stencil=%u",
                  p,
                  ri->colorAttachmentCount,
                  ri->depthAttachmentFormat,
                  ri->stencilAttachmentFormat);
                 rw->viewMask = 0x3;
                }
                view_mask = rw->viewMask;
                STEREO_LOG_VERBOSE(
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
            STEREO_LOG_VERBOSE(
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
            STEREO_LOG_VERBOSE("PIPE_INVALID p=%u ci=%p stageCount=%u pStages=%p renderPass=%p",
                       p,
                       (void*)ci,
                       ci ? ci->stageCount : 0,
                       ci ? (void*)ci->pStages : NULL,
                       ci ? (void*)ci->renderPass : NULL);
            continue;
        }
        bool has_vs=false, has_tcs=false, has_tes=false, has_gs=false;
        uint32_t vs_stage=~0u, tes_stage=~0u;
        for (uint32_t s=0;s<ci->stageCount;s++) {
            VkShaderStageFlagBits st=ci->pStages[s].stage;
            if (st==VK_SHADER_STAGE_VERTEX_BIT)
                { has_vs=true; vs_stage=s; }
            if (st==VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT)
                has_tcs=true;
            if (st==VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)
                { has_tes=true; tes_stage=s; }
            if (st==VK_SHADER_STAGE_GEOMETRY_BIT)
                has_gs=true;
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
            "PIPE_DECISION p=%u rp=%p in_mv_rp=%d viewMask=0x%x stages=%u has_vs=%d has_tcs=%d has_tes=%d has_gs=%d",
            p,
            (void*)ci->renderPass,
            (int)in_mv_rp,
            (unsigned)view_mask,
            ci->stageCount,
            (int)has_vs,
            (int)has_tcs,
            (int)has_tes,
            (int)has_gs);
        for (uint32_t fs_dbg_i = 0; fs_dbg_i < ci->stageCount; fs_dbg_i++) {
            if (ci->pStages[fs_dbg_i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                StereoShaderCache *fs_dbg =
                    cache_find(sd, ci->pStages[fs_dbg_i].module);
                if (fs_dbg) {
                    STEREO_LOG_VERBOSE(
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
        /* ── PATCH 3: Pipeline multiview state ─────────────────────────────────── */
        /* NOTE: VkGraphicsPipelineCreateInfo.pMultiviewState does not exist in
         * Vulkan 1.4 SDK headers (removed in favour of render-pass-driven MV).
         * Pipeline multiview is inherited from the MV render pass (viewMask=0x3)
         * created in render_pass.c.  The key requirement is that the DEVICE
         * must have the multiview feature enabled — see device.c where we
         * also patch VkPhysicalDeviceVulkan11Features.multiview=VK_TRUE. */
        if (in_mv_rp) {
            if (rpi && rpi->mv_handle) {
                STEREO_LOG(
                    "PIPE_RP_SWAP p=%u orig_rp=%p mv_rp=%p has_mv=%u — swapping renderPass for multiview",
                    p,
                    (void*)ci->renderPass,
                    (void*)rpi->mv_handle,
                    (unsigned)rpi->has_multiview);
                /* render-pass pipeline path only */
                infos[p].renderPass = rpi->mv_handle;
            } else {
                STEREO_LOG(
                    "PIPE_RP_NOSWAP p=%u orig_rp=%p rpi=%p mv_handle=%p has_mv=%u — NOT swapping (dynamic rendering or missing mv_handle)",
                    p,
                    (void*)ci->renderPass,
                    (void*)rpi,
                    rpi ? (void*)rpi->mv_handle : NULL,
                    rpi ? (unsigned)rpi->has_multiview : 0);
                /* VK 1.3 dynamic rendering: keep infos[p].renderPass as-is */
            }
        }
        if (!in_mv_rp)
        {
            STEREO_LOG_VERBOSE(
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
        /* ── Debug FS: red/green ViewIndex test ──────────────────────────
         * When VKS3D_DEBUG_VIEW=1, replace FS with a solid-colour shader:
         * ViewIndex=0 → red, ViewIndex=1 → green. Skip VS patching entirely
         * so geometry is identical between eyes; the ONLY variable is the
         * FS colour, giving a clean ViewIndex diagnostic.               */
        if (in_mv_rp && is_debug_view_enabled())
        {
            uint32_t fs_s = ~0u;
            for (uint32_t s2 = 0; s2 < ci->stageCount; s2++)
            {
                if (ci->pStages[s2].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
                { fs_s = s2; break; }
            }
            if (fs_s != ~0u)
            {
                uint32_t *dbg_spv = NULL; size_t dbg_wc = 0;
                if (build_debug_fs_spv(&dbg_spv, &dbg_wc))
                {
                    VkShaderModuleCreateInfo mci = {
                        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0,
                        dbg_wc * 4, dbg_spv
                    };
                    VkShaderModule dbg_mod = VK_NULL_HANDLE;
                    VkResult drc = sd->real.CreateShaderModule(
                        sd->real_device, &mci, NULL, &dbg_mod);
                    free(dbg_spv);
                    if (drc == VK_SUCCESS && dbg_mod != VK_NULL_HANDLE)
                    {
                        if (!tst[p])
                        {
                            tst[p] = malloc(ci->stageCount * sizeof(*tst[0]));
                            memcpy(tst[p], ci->pStages,
                                   ci->stageCount * sizeof(*tst[0]));
                            infos[p].pStages = tst[p];
                            infos[p].stageCount = ci->stageCount;
                        }
                        tst[p][fs_s].module = dbg_mod;
                        tmp_mod[p] = dbg_mod; /* track for cleanup */
                        STEREO_LOG(
                            "DEBUG_VIEW_FS p=%u fs_s=%u module=%p — red/green injected",
                            p, fs_s, (void*)dbg_mod);
                        goto PIPE_DECISION_CONTINUE; /* skip VS/TES patching */
                    }
                    else
                    {
                        STEREO_LOG(
                            "DEBUG_VIEW_FS p=%u FAILED rc=%d — falling through to normal",
                            p, (int)drc);
                    }
                }
            }
            else
            {
                STEREO_LOG("DEBUG_VIEW_FS p=%u no FS stage found — skipping", p);
            }
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
        /* Check for VK_DYNAMIC_STATE_VERTEX_INPUT_EXT (value 93).
         * Yuzu may use dynamic vertex input or vertex pulling (reading
         * vertex data from SSBO), so pVertexInputState is NULL but the
         * pipeline is NOT a fullscreen quad — it renders geometry.
         * We no longer skip VS patching for "quad" pipelines; instead
         * we always try VS patching. spirv_patch_stereo_vertex will
         * check for a projection matrix and skip if not found, which
         * correctly handles true fullscreen-quad VS (no projection). */
        bool is_quad = pipe_is_quad_pipeline(ci);
        STEREO_LOG_VERBOSE(
            "FS_GATE p=%u quad=%u stageCount=%u",
            p,
            is_quad,
            ci->stageCount);
        /* FS patch for full-screen quad pipelines (plus any other pipelines
         * that render into multiview passes with FS that samples from
         * stereo-upgraded attachments).
         *
         * Problem: 3D pipelines (with vertex input) write to 2-layer intermediate
         * render targets (via multiview — layer 0 = left eye, layer 1 = right eye).
         * Full-screen-quad pipelines (deferred lighting, blit passes, TAA, etc.)
         * sample those attachments.  Before patching, the FS uses sampler2D and
         * always reads layer 0 (implicitly).  So both stereo output layers end up
         * with the left-eye-only G-buffer content → left/right identical.
         *
         * Fix: in multiview render passes, patch every FS:
         *   1. Change every relevant OpTypeImage from sampler2D → sampler2DArray
         *      (the image view was already upgraded to 2D_ARRAY by stereo_CreateImageView
         *       for any stereo-tracked image, so Vulkan type matching is safe).
         *   2. Rewrite every OpImageSampleImplicitLod / ExplicitLod / Dref / Gather
         *      that uses those images: vec2 uv → vec3(uv, gl_ViewIndex).
         *      Each eye then reads its own G-buffer layer.
         *
         * Geometry pipelines still get Path A (TES) / Path B (VS) patching for
         * the per-vertex stereo offset (projection-mode parallax).  Both patches
         * are independent and both must be applied for the full pipeline to work. */
        /* FS patching is DISABLED.
         *
         * Root cause of crash: fs_binding_is_stereo_attachment() uses a
         * "binding <= 4" heuristic to decide which sampler2D to convert
         * to sampler2DArray.  This is far too broad — material textures,
         * lookup tables and other non-stereo resources commonly use
         * bindings 0-4.  Converting them to sampler2DArray while the
         * bound image view is actually 2D (not 2D_ARRAY) produces invalid
         * SPIR-V that crashes the NVIDIA driver inside
         * vkCreateGraphicsPipelines.
         *
         * The proper fix requires cross-referencing each FS sampler's
         * (set, binding) against the device's intercepted_color tracking
         * list (images VKS3D actually upgraded to 2D_ARRAY).  Until that
         * plumbing is implemented, FS patching must stay disabled to
         * avoid the crash.
         *
         * VS patching (Path B) still handles per-vertex stereo parallax
         * for geometry pipelines.  Post-process quad pipelines will read
         * layer 0 only (left eye), so post-processed frames may show
         * identical left/right content — but the core 3D scene geometry
         * still has correct parallax. */
        if (false && in_mv_rp && ci->stageCount > 0 && is_quad)
        {
            STEREO_LOG(
                "FS_PATCH_ENTRY pipe=%u rp=%p stageCount=%u",
                p,
                (void*)ci->renderPass,
                ci->stageCount);
            /* Find FS stage */
            uint32_t fs_s = ~0u;
            for (uint32_t s2 = 0; s2 < ci->stageCount; s2++)
            {
                if (ci->pStages[s2].stage == VK_SHADER_STAGE_FRAGMENT_BIT)
                {
                    fs_s = s2;
                    break;
                }
            }
            STEREO_LOG(
                "FS_PATCH_FS_STAGE pipe=%u fs_s=%u",
                p,
                (fs_s == ~0u) ? 0xffffffffu : fs_s);
            if (fs_s == ~0u)
            {
                STEREO_LOG_VERBOSE("Pipe %u: quad but no FS stage", p);
                continue;
            }
            StereoShaderCache *fs_cache =
                cache_find(sd, ci->pStages[fs_s].module);
            STEREO_LOG(
                "FS_PATCH_CACHE pipe=%u module=%p cache=%p",
                p,
                (void*)ci->pStages[fs_s].module,
                (void*)fs_cache);
            if (!fs_cache)
            {
                STEREO_LOG_VERBOSE(
                    "PIPE_MODULE_MISS stage=0x%x module=%p renderPass=%p pipeline=%u",
                    ci->pStages[fs_s].stage,
                    (void *)ci->pStages[fs_s].module,
                    (void *)ci->renderPass,
                    p);
                for (uint32_t k = 0; k < sd->shader_cache_count; ++k)
                {
                    STEREO_LOG_VERBOSE(
                        "CACHE_HANDLE[%u] module=%p hash=%016llx words=%zu",
                        k,
                        (void *)sd->shader_cache[k].handle,
                        (unsigned long long)hash_spv(
                            sd->shader_cache[k].spv,
                            sd->shader_cache[k].words),
                        sd->shader_cache[k].words);
                }
                continue;
            }
            uint64_t spv_hash =
                hash_spv(fs_cache->spv, fs_cache->words);
            uint32_t pipeline_has_mv =
                (rpi != NULL) ? (uint32_t)rpi->has_multiview : 0;
            STEREO_LOG_VERBOSE(
                "FS_PATCH_DECISION "
                "pipe=%u "
                "rp=%p "
                "subpass=%u "
                "is_quad=%u "
                "pipeline_mv=%u "
                "has_fs=%u "
                "fs_hash=%016llx "
                "cache=%p "
                "stageFlags=0x%x",
                p,
                (void *)ci->renderPass,
                ci->subpass,
                is_quad,
                pipeline_has_mv,
                (fs_s != ~0u),
                (unsigned long long)spv_hash,
                (void *)fs_cache,
                ci->pStages[fs_s].stage);
            STEREO_LOG_VERBOSE(
                "QUAD_FS_SHADER p=%u hash=%016llx words=%zu module=%p",
                p,
                (unsigned long long)spv_hash,
                fs_cache->words,
                (void *)ci->pStages[fs_s].module);
            STEREO_LOG_VERBOSE(
                "SHADER_MODULE stage=FS hash=%016llx words=%zu module=%p",
                (unsigned long long)spv_hash,
                fs_cache->words,
                (void *)ci->pStages[fs_s].module);
            STEREO_LOG_VERBOSE(
                "PATCH hash=%016llx words=%zu module=%p vs_stage=%u",
                (unsigned long long)spv_hash,
                fs_cache->words,
                (void *)(has_vs ? ci->pStages[vs_stage].module : VK_NULL_HANDLE),
                vs_stage);
            if (dump)
            {
                char dp[512];

                _snprintf(
                    dp,
                    sizeof(dp) - 1,
                    "%s\\%016llx-fs.spv",
                    dump,
                    (unsigned long long)spv_hash);

                FILE *f = fopen(dp, "rb");
                if (!f)
                {
                    f = fopen(dp, "wb");
                    if (f)
                    {
                        fwrite(
                            fs_cache->spv,
                            4,
                            fs_cache->words,
                            f);
                        fclose(f);
                    }
                }
                else
                {
                    fclose(f);
                }
            }
            uint32_t *patched = NULL; size_t pc2 = 0;
            STEREO_LOG_VERBOSE(
                "FS_PATCH_BEGIN hash=%016llx",
                (unsigned long long)spv_hash);
            STEREO_LOG_VERBOSE(
                "PATCHING_FS hash=%016llx",
                (unsigned long long)spv_hash);
            STEREO_LOG_VERBOSE(
                "CALLING spirv_patch_stereo_fs hash=%016llx words=%zu",
                (unsigned long long)spv_hash,
                fs_cache->words);
            /*
             * Analyze FS projection UBO usage.
             *
             * SSAO/reconstruction shaders often use the projection matrix
             * only in the fragment stage, so VS metadata is insufficient.
             */
            {
                SpvMod fm = {0};
                fm.words = fs_cache->spv;
                fm.count = fs_cache->words;
                /* Guard against truncated/invalid SPIR-V modules:
                 * words[3] is the Bound ID.  Without >=4 words the
                 * header is incomplete and any access to words[3]
                 * would read out of bounds. */
                if (fs_cache->words < 4)
                {
                    STEREO_LOG(
                        "FS_PATCH_SKIP_SHORT pipe=%u words=%zu (need >=4)",
                        p,
                        fs_cache->words);
                    continue;
                }
                fm.bound = fm.words[3];
                fm.value_capacity = fm.bound + 64;
                fm.value_from_matrix =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.is_matrix_type =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.is_matrix_ptr =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.is_proj_value =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.is_view_value =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.is_io_value =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.is_io_ptr =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.ac_base =
                    calloc(fm.value_capacity, sizeof(uint32_t));
                fm.is_const_value =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                fm.vec4_w_id =
                    calloc(fm.value_capacity, sizeof(uint32_t));
                fm.mtv_vec_w_const =
                    calloc(fm.value_capacity, sizeof(uint8_t));
                if (fm.value_from_matrix &&
                    fm.is_matrix_type &&
                    fm.is_matrix_ptr &&
                    fm.is_proj_value &&
                    fm.is_view_value &&
                    fm.is_io_value &&
                    fm.is_io_ptr &&
                    fm.ac_base &&
                    fm.is_const_value &&
                    fm.vec4_w_id &&
                    fm.mtv_vec_w_const)
                {
                    spv_scan(&fm);
                    if (fm.proj_found)
                    {
                        dbg_out[p].has_proj_ubo = true;
                        dbg_out[p].proj_set = fm.proj_set;
                        dbg_out[p].proj_binding = fm.proj_binding;
                        dbg_out[p].proj_member_mask =
                            fm.proj_member_mask;
                        dbg_out[p].proj_var = fm.proj_var;
                        STEREO_LOG_VERBOSE(
                            "FS_PROJ_FOUND hash=%016llx set=%u binding=%u mask=0x%X var=%u",
                            (unsigned long long)hash_spv(
                                fs_cache->spv,
                                fs_cache->words),
                            fm.proj_set,
                            fm.proj_binding,
                            fm.proj_member_mask,
                            fm.proj_var);
                    }
                }
                free_spv_provenance(&fm);
            }
            STEREO_LOG(
                "FS_PATCH_PREP_DONE pipe=%u hash=%016llx words=%zu bound=%u",
                p,
                (unsigned long long)spv_hash,
                fs_cache->words,
                fs_cache->words >= 4 ? fs_cache->spv[3] : 0);
            STEREO_LOG(
                "FS_PATCH_BEGIN pipe=%u hash=%016llx words=%zu",
                p,
                (unsigned long long)spv_hash,
                fs_cache->words);
            if (!spirv_patch_stereo_fs(
                    fs_cache->spv,
                    fs_cache->words,
                    &patched,
                    &pc2))
            {
                STEREO_LOG(
                    "Pipe %u: FS patch skipped (no 2D samplers for stereo attachments)",
                    p);
                continue;
            }
            STEREO_LOG(
                "FS_PATCH_DONE pipe=%u hash=%016llx words_in=%zu words_out=%zu",
                p,
                (unsigned long long)spv_hash,
                fs_cache->words,
                pc2);
            STEREO_LOG_VERBOSE(
                "spirv_patch_stereo_fs returned=%u patchedWords=%zu",
                patched ? 1 : 0,
                pc2);
            STEREO_LOG_VERBOSE(
                "FS_PATCH_RETURN ptr=%p words=%zu hash=%016llx",
                (void *)patched,
                pc2,
                (unsigned long long)hash_spv(
                    patched,
                    pc2));
            STEREO_LOG_VERBOSE(
                "FS_DUMP words=%zu ptr=%p hash=%016llx",
                pc2,
                (void *)patched,
                (unsigned long long)hash_spv(
                    patched,
                    pc2));
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
            STEREO_LOG_VERBOSE(
                "PATCHED_STAGE PathFS p=%u stage=%u orig=%p patched=%p",
                p,
                fs_s,
                (void *)ci->pStages[fs_s].module,
                (void *)tmp);
            STEREO_LOG_VERBOSE(
                "Pipe %u: Path FS — quad sampler2DArray patch (%u stages)",
                p,
                sc2);
            continue;
        }
        /* ── Path A: patch existing TES ──────────────────────────────── */
        if (has_tes && tes_stage!=~0u) {
            StereoShaderCache *e=cache_find(sd, ci->pStages[tes_stage].module);
            if (!e) { STEREO_LOG_VERBOSE("Pipe %u PathA: TES not cached",p); continue; }
            STEREO_LOG_VERBOSE(
                "SHADER_MODULE stage=TES hash=%016llx words=%zu module=%p",
                (unsigned long long)hash_spv(e->spv, e->words),
                e->words,
                (void*)ci->pStages[tes_stage].module);
            if (dump)
            {
                uint64_t spv_hash=hash_spv(e->spv, e->words);
                char dp[512];
                _snprintf(
                    dp,
                    sizeof(dp)-1,
                    "%s\\%016llx-ts.spv",
                    dump,
                    (unsigned long long)spv_hash);
                FILE *f = fopen(dp, "rb");
                if (!f)
                {
                    f = fopen(dp, "wb");
                    if (f)
                    {
                        fwrite(
                            e->spv,
                            4,
                            e->words,
                            f);
                        fclose(f);
                    }
                }
                else
                {
                    fclose(f);
                }
            }
            uint32_t *patched=NULL; size_t pc2=0;
            STEREO_LOG_VERBOSE(
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
                    pipe_vtx_binding_count(ci),
                    pipe_is_quad_pipeline(ci) ? 1u : 0u
                };
                if (!spirv_patch_stereo_vertex(
                        &sd->stereo,
                        e->spv, e->words,
                        &patched, &pc2,
                        lo, ro, conv,
                        true,
                        &dbgA))
                {
                STEREO_LOG_VERBOSE("TES patch failed");
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
                STEREO_LOG_VERBOSE("Pipe %u PathA: patch failed",p);
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
            STEREO_LOG_VERBOSE(
                "PATCHED_STAGE PathA p=%u stage=%u orig=%p patched=%p",
                p,
                tes_stage,
                (void *)ci->pStages[tes_stage].module,
                (void *)tmp);
            STEREO_LOG_VERBOSE(
                "Pipe %u: Path A — TES patched (gl_ViewIndex)",
                p);
            continue;
        }
        STEREO_LOG(
            "PATHB_GATE p=%u in_mv_rp=%d has_vs=%d has_tcs=%d has_gs=%d vs_stage=%u",
            p,
            (int)in_mv_rp,
            (int)has_vs,
            (int)has_tcs,
            (int)has_gs,
            (unsigned)vs_stage);
        /* ── Path B: VS-only pipeline → directly patch VS itself (inject ViewIndex + OFF_AXIS)
         * See file header (per user's corrected translation) — THIS is the correct modern path:
         *   • "路径B — 基于VS的管线（无原生曲面细分）：直接修改VS顶点着色器，注入 gl_ViewIndex。"
         *   • "只要驱动完整实现 VK_KHR_multiview 扩展即可生效（目前所有英伟达、AMD、Intel驱动均支持）。"
         *   • "该方案替代了旧的 TCS+TES 注入逻辑；旧方案原本只是针对426.06版本驱动的临时规避手段，
         *      在新版驱动上，由于PerVertex块校验规则变得严格，会发生接口不匹配导致程序崩溃。"
         * → we MUST NOT synthesise any extra TCS/TES/GS stages anymore (those are obsolete
         *   and cause black screens / driver crashes).  Replace ONLY the VS stage's module
         *   with a patched copy that reads gl_ViewIndex and applies the stereo offset.
         *   Everything else (stageCount, tessellationState, topology, draw commands)
         *   stays completely unchanged.                                       */
        if (in_mv_rp &&
            ci->stageCount > 0 &&
            has_vs &&
            !has_tcs &&
            !has_gs &&
            vs_stage != ~0u) {
            StereoShaderCache *e=cache_find(sd, ci->pStages[vs_stage].module);
            if (!e) { STEREO_LOG_VERBOSE("Pipe %u PathB: VS not cached",p); continue; }
            uint32_t *patched = NULL; size_t pc = 0;
            StereoDebugCtx dbgB;
            dbgB.pipeline_index         = p;
            dbgB.render_pass            = ci->renderPass;
            dbgB.is_multiview           = in_mv_rp;
            dbgB.stage                  = (uint32_t)VK_SHADER_STAGE_VERTEX_BIT;
            dbgB.vertex_binding_count   = pipe_vtx_binding_count(ci);
            dbgB.is_quad                = pipe_is_quad_pipeline(ci) ? 1u : 0u;
            dbgB.has_proj_ubo           = VK_FALSE;
            dbgB.proj_set               = UINT32_MAX;
            dbgB.proj_binding           = UINT32_MAX;
            dbgB.proj_member_mask       = UINT32_MAX;
            dbgB.proj_var               = UINT32_MAX;
            dbgB.has_matrix_ops         = false;
            dbgB.direct_position_write  = false;
            STEREO_LOG(
                "PATHB_VS p=%u vs_words=%zu — patching VS directly (no synth stages; gl_ViewIndex in VS per header)",
                p, e->words);
            if (!spirv_patch_stereo_vertex(
                    &sd->stereo,
                    e->spv, e->words,
                    &patched, &pc,
                    lo, ro, conv,
                    /*inj_vi=*/true,
                    &dbgB)) {
                STEREO_LOG("Pipe %u PathB: VS patch failed — not patched", p);
                continue;
            }
            dbg_out[p] = dbgB;
            STEREO_LOG("PATHB_VS_DONE p=%u words_in=%zu words_out=%zu", p, e->words, pc);
            VkShaderModule vs_mod = VK_NULL_HANDLE;
            {
                VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                smci.codeSize = pc * 4;
                smci.pCode    = patched;
                VkResult mr = sd->real.CreateShaderModule(sd->real_device, &smci, NULL, &vs_mod);
                spirv_patched_free(patched);
                if (mr != VK_SUCCESS) {
                    STEREO_ERR("Pipe %u PathB: patched VS module err %d", p, (int)mr);
                    continue;
                }
            }
            /* Replace ONLY the VS stage's module. All other stages, stageCount,
             * tessellationState, input assembly, dynamic state etc. are kept
             * exactly as the game intended — draw commands remain valid. */
            uint32_t sc = ci->stageCount;
            VkPipelineShaderStageCreateInfo *st = malloc(sc * sizeof(*st));
            if (!st) {
                sd->real.DestroyShaderModule(sd->real_device, vs_mod, NULL);
                continue;
            }
            memcpy(st, ci->pStages, sc * sizeof(*st));
            st[vs_stage].module = vs_mod;
            infos[p].pStages    = st;
            /* leave infos[p].stageCount unchanged (still ci->stageCount) */
            /* leave infos[p].pTessellationState unchanged (usually NULL) */
            tmp_mod[p]     = vs_mod;     /* patched VS module — registered for cleanup */
            tmp_mod_tcs[p] = VK_NULL_HANDLE;
            tst[p]         = st;
            STEREO_LOG(
                "Pipe %u: Path B — original VS @ stage[%u] replaced with patched(%p) stages=%u (NO synth stages; Draw/DrawIndexed 100%% compatible)",
                p, vs_stage, (void*)vs_mod, (unsigned)sc);
            continue;
        }
        STEREO_LOG_VERBOSE("Pipe %u: no patchable VS/TES stage (stageCount=%u has_vs=%d has_tes=%d has_tcs=%d has_gs=%d) — not patched",
                   p, ci->stageCount, has_vs, has_tes, has_tcs, has_gs);
    }
    PIPE_DECISION_CONTINUE:
    /* ── PATCH 5: RenderPass-based multiview binding ─────────────── */
    for (uint32_t p = 0; p < N; p++) {
        StereoRenderPassInfo *rpi = NULL;
        if (pCI[p].renderPass != VK_NULL_HANDLE)
            rpi = stereo_rp_lookup(sd, pCI[p].renderPass);
        STEREO_LOG_VERBOSE(
            "PIPE_RP p=%u ci_rp=%p rpi=%p has_mv=%u mv=%p",
            p,
            (void*)pCI[p].renderPass,
            (void*)rpi,
            rpi ? (unsigned)rpi->has_multiview : 0,
            rpi ? (void*)rpi->mv_handle : NULL);
        if (rpi && rpi->has_multiview) {
            STEREO_LOG_VERBOSE("Pipe %u: binding MV render pass %p", p, (void*)rpi->mv_handle);
            infos[p].renderPass = rpi->mv_handle;
        }
    }
    for (uint32_t p = 0; p < N; p++) {
        STEREO_LOG_VERBOSE(
            "PIPE_FINAL p=%u ci_rp=%p final_rp=%p stages=%u",
            p,
            (void*)pCI[p].renderPass,
            (void*)infos[p].renderPass,
            infos[p].stageCount);
    }
    for (uint32_t p = 0; p < N; ++p)
    {
        STEREO_LOG_VERBOSE(
            "PIPE_CREATE pipeline=%u renderPass=%p subpass=%u",
            p,
            infos[p].renderPass,
            infos[p].subpass);
        for (uint32_t s = 0; s < infos[p].stageCount; s++)
        {
            STEREO_LOG_VERBOSE(
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
    STEREO_LOG_VERBOSE(
        "[PIPE BEFORE DRIVER] N=%u",
        N);
    for (uint32_t p = 0; p < N; ++p)
    {
        STEREO_LOG(
            "[PIPE %u] patched=%d tmp=%p stages=%u rp=%p (orig_rp=%p)",
            p,
            tst[p] != NULL,
            (void*)tmp_mod[p],
            infos[p].stageCount,
            (void*)infos[p].renderPass,
            (void*)pCI[p].renderPass);
        for (uint32_t s = 0; s < infos[p].stageCount; ++s)
        {
            const VkPipelineShaderStageCreateInfo *st =
                &infos[p].pStages[s];
            STEREO_LOG_VERBOSE(
                "    stage=%u module=%p stageFlags=0x%x",
                s,
                (void*)st->module,
                st->stage);
        }
    }
    VkResult res=sd->real.CreateGraphicsPipelines(sd->real_device,pc,N,infos,pAlloc,pP);
    STEREO_LOG(
        "PIPE_CREATE_RESULT res=%d N=%u",
        (int)res,
        (unsigned)N);
    for (uint32_t p = 0; p < N; p++) {
        if (pP[p] == VK_NULL_HANDLE) {
            STEREO_LOG("PIPE_NULL p=%u handle=NULL patched=%d",
                       p, (int)(tmp_mod[p] != VK_NULL_HANDLE));
        }
    }
    for (uint32_t p = 0; p < N; p++) {
        /* Accumulate per-pipe summary stats once */
        bool pipe_is_quad = pipe_is_quad_pipeline(&pCI[p]);
        bool has_any_stage = (pCI[p].stageCount > 0);
        bool mv_detected_here = false;
        {
            StereoRenderPassInfo *rpi2 = NULL;
            if (pCI[p].renderPass != VK_NULL_HANDLE) {
                rpi2 = stereo_rp_lookup(sd, pCI[p].renderPass);
                if (rpi2 && rpi2->has_multiview) mv_detected_here = true;
            }
            if (sd->stereo.multiview) {
                const VkBaseInStructure *b = (const VkBaseInStructure*)pCI[p].pNext;
                while (b) {
                    if (b->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
                        const VkPipelineRenderingCreateInfo *ri2 =
                            (const VkPipelineRenderingCreateInfo*)b;
                        if ((ri2->viewMask & 0x3) != 0) mv_detected_here = true;
                    }
                    b = b->pNext;
                }
            }
        }
        bool patched_tmp_exists = (tmp_mod[p] != VK_NULL_HANDLE);
        InterlockedIncrement((volatile long*)&g_stat_pipes_total);
        if (mv_detected_here)
            InterlockedIncrement((volatile long*)&g_stat_pipes_mv);
        else
            InterlockedIncrement((volatile long*)&g_stat_pipes_mono);
        if (res == VK_SUCCESS && patched_tmp_exists)
            InterlockedIncrement((volatile long*)&g_stat_pipes_patch_ok);
        else if (res == VK_SUCCESS)
            InterlockedIncrement((volatile long*)&g_stat_pipes_patch_fail);
        STEREO_LOG(
            "PIPE_SUMMARY p=%u result=%d mv=%u quad=%u stages=%u patched=%u "
            "rp=%p proj_info=%s view_mask_checked=%u",
            p,
            (int)res,
            (unsigned)mv_detected_here,
            (unsigned)pipe_is_quad,
            (unsigned)infos[p].stageCount,
            (unsigned)patched_tmp_exists,
            (void*)infos[p].renderPass,
            (dbg_out[p].has_proj_ubo ? "YES" : "NO"),
            (unsigned)dbg_out[p].proj_member_mask);
        /* Diagnostic: why is a pipeline classified as quad? */
        if (pipe_is_quad)
        {
            bool has_dyn_vtx = pipe_has_dynamic_vtx_input(&pCI[p]);
            uint32_t bind_cnt = 0;
            const char *vis_ptr = "NULL";
            if (pCI[p].pVertexInputState)
            {
                bind_cnt = pCI[p].pVertexInputState->vertexBindingDescriptionCount;
                vis_ptr = "OK";
            }
            STEREO_LOG(
                "PIPE_QUAD_DIAG p=%u dyn_vtx=%u vis=%s bind_cnt=%u "
                "pDynState=%p dynCount=%u",
                p,
                (unsigned)has_dyn_vtx,
                vis_ptr,
                bind_cnt,
                (void*)pCI[p].pDynamicState,
                pCI[p].pDynamicState ?
                    pCI[p].pDynamicState->dynamicStateCount : 0);
        }
        STEREO_LOG_VERBOSE(
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
                info->is_quad = pipe_is_quad_pipeline(&pCI[p]);

                info->vertex_binding_count = pipe_vtx_binding_count(&pCI[p]);
                info->view_mask = 0; /* default */
                info->has_proj_ubo          = dbg_out[p].has_proj_ubo;
                info->proj_set              = dbg_out[p].proj_set;
                info->proj_binding          = dbg_out[p].proj_binding;
                info->proj_member_mask      = dbg_out[p].proj_member_mask;
                info->proj_var              = dbg_out[p].proj_var;
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
                        /*
                         * Always probe the fragment shader too.
                         * VS/TES may already have filled proj info, but FS can carry
                         * its own projection UBO for SSAO / post-process paths.
                         */
                        StereoShaderCache *fs_cache = cache_find(sd, st->module);
                        if (fs_cache)
                        {
                            uint64_t h = hash_spv(fs_cache->spv, fs_cache->words);
                            STEREO_LOG_VERBOSE(
                                "FS_PIPE_MODULE module=%p hash=%016llx words=%zu",
                                (void *)st->module,
                                (unsigned long long)h,
                                fs_cache->words);
                        }
                    }
                }
            }
            STEREO_LOG_VERBOSE(
                "PIPE_INFO pipe=%p rp=%p orig_rp=%p stages=%u",
                (void*)pP[p],
                (void*)infos[p].renderPass,
                (void*)pCI[p].renderPass,
                infos[p].stageCount);
        }
    }
    STEREO_LOG_VERBOSE(
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
        if (tmp_mod_tcs[p]) {
            if (sd->tmp_module_count<MAX_TMP_MODULES)
                sd->tmp_modules[sd->tmp_module_count++]=tmp_mod_tcs[p];
            else
                sd->real.DestroyShaderModule(sd->real_device,tmp_mod_tcs[p],NULL);
        }
        free(tst[p]);
    }
    free(tmp_mod); free(tmp_mod_tcs); free(tst); free(infos); free(tess_infos); free(dbg_out);
    STEREO_LOG("PIPE_CREATE_DONE N=%u result=%d", N, (int)res);
    return res;
}

/* ── vkDestroyShaderModule ───────────────────────────────────────────────── */
VKAPI_ATTR void VKAPI_CALL
stereo_DestroyShaderModule(VkDevice device, VkShaderModule sm,
                           const VkAllocationCallbacks *pAlloc)
{
    STEREO_LOG_ONCE("FIRST_CALL stereo_DestroyShaderModule module=%p", (void*)sm);
    STEREO_LOG_VERBOSE("CALLED stereo_DestroyShaderModule");
    StereoDevice *sd=stereo_device_from_handle(device);
    if (!sd) return;
    cache_remove(sd,sm);
    sd->real.DestroyShaderModule(sd->real_device,sm,pAlloc);
}
