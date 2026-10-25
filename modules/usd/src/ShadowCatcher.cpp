// Copyright (c) 2026 jesus luque.
#include "athenea/usd/ShadowCatcher.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include <pxr/base/gf/range3d.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/bboxCache.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include "athenea/core/Log.h"

namespace athenea::usd {

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

/// The stage's up axis as an index: 2 for Z, 1 for Y.
int upIndex(const UsdStageRefPtr& stage) {
    return UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z ? 2 : 1;
}

bool under(const SdfPath& path, const SdfPath& root) {
    return path == root || path.HasPrefix(root);
}

}   // namespace

Result<ShadowCatcherStage> writeShadowCatcherStage(const std::filesystem::path& source,
                                                   const ShadowCatcherOptions& options,
                                                   const std::filesystem::path& into) {
    UsdStageRefPtr stage = UsdStage::Open(source.string());
    if (!stage) {
        return Error::make(ErrorCode::IoFailure, "shadow catcher: cannot open {}", source.string());
    }
    const SdfPath objectPath(options.object.empty() ? std::string("/") : options.object);
    const UsdPrim object = stage->GetPrimAtPath(objectPath);
    if (!object) {
        return Error::make(ErrorCode::NotFound, "shadow catcher: no prim {} in {}", objectPath.GetString(),
                           source.string());
    }
    // The extents the file carries, with their hints: what a box is read
    // from, not the points.
    UsdGeomBBoxCache cache(UsdTimeCode::Default(), {UsdGeomTokens->default_, UsdGeomTokens->render}, true);
    const GfRange3d objectBox = cache.ComputeWorldBound(object).ComputeAlignedRange();
    if (objectBox.IsEmpty()) {
        return Error::make(ErrorCode::InvalidArgument, "shadow catcher: {} has no extent to cast from",
                           objectPath.GetString());
    }
    const int up = upIndex(stage);
    const double height = objectBox.GetMax()[up] - objectBox.GetMin()[up];
    const double bottom = objectBox.GetMin()[up];

    // THE GROUND: named, or the largest flat mesh outside the object whose
    // top is where the object's bottom is (within a tenth of its height) and
    // which reaches under it.
    UsdPrim ground;
    if (!options.ground.empty()) {
        ground = stage->GetPrimAtPath(SdfPath(options.ground));
        if (!ground) {
            return Error::make(ErrorCode::NotFound, "shadow catcher: no ground prim {}", options.ground);
        }
    } else {
        double widest = 0.0;
        for (const UsdPrim& prim : stage->Traverse()) {
            if (!prim.IsA<UsdGeomMesh>() || under(prim.GetPath(), objectPath)) {
                continue;
            }
            const GfRange3d box = cache.ComputeWorldBound(prim).ComputeAlignedRange();
            if (box.IsEmpty()) {
                continue;
            }
            const GfVec3d size = box.GetSize();
            const int a = 0;
            const int b = up == 2 ? 1 : 2;
            const double area = size[a] * size[b];
            const double thick = size[up];
            const double across = std::max(size[a], size[b]);
            const bool flat = thick <= 0.02 * across;
            const bool touching = std::abs(box.GetMax()[up] - bottom) <= 0.1 * height;
            const bool beneath = box.GetMin()[a] <= objectBox.GetMin()[a] && box.GetMax()[a] >= objectBox.GetMax()[a] &&
                                 box.GetMin()[b] <= objectBox.GetMin()[b] && box.GetMax()[b] >= objectBox.GetMax()[b];
            if (flat && touching && beneath && area > widest) {
                widest = area;
                ground = prim;
            }
        }
        if (!ground) {
            return Error::make(ErrorCode::NotFound,
                               "shadow catcher: no ground under {} (a flat mesh outside it, at its bottom, "
                               "reaching under it); name one with --catcher-ground",
                               objectPath.GetString());
        }
    }
    const GfRange3d groundBox = cache.ComputeWorldBound(ground).ComputeAlignedRange();
    const double level = groundBox.IsEmpty() ? bottom : groundBox.GetMax()[up];

    // THE PATCH: the object's footprint widened by `margin` heights, held to
    // the ground's own, a hair above it so the bake's rays leave the patch
    // rather than the ground.
    const int a = 0;
    const int b = up == 2 ? 1 : 2;
    // Along the object's longer horizontal axis a light catcher reaches
    // further: `reachLong` of its lengths past either end.
    const int along = objectBox.GetSize()[a] >= objectBox.GetSize()[b] ? a : b;
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};
    for (const int k : {a, b}) {
        const double reach = k == along && options.reachLong > 0.0
                                 ? std::max(options.margin * height, options.reachLong * objectBox.GetSize()[k])
                                 : options.margin * height;
        lo[k] = objectBox.GetMin()[k] - reach;
        hi[k] = objectBox.GetMax()[k] + reach;
        if (!groundBox.IsEmpty()) {
            lo[k] = std::max(lo[k], groundBox.GetMin()[k]);
            hi[k] = std::min(hi[k], groundBox.GetMax()[k]);
        }
    }
    const double lift = 1.0e-4 * std::max(height, 1.0e-6);
    lo[up] = hi[up] = level + lift;

    // A GRID OF QUADS, about 64 cells a side each: a conversion walks a
    // triangle's cells up to --max-cells, and two triangles over the
    // Corvette's 6 x 8 m left a third of the patch unsampled.
    const double cell = options.cell > 0.0 ? options.cell : height / std::max(options.cellsPerHeight, 1.0);
    const double across = std::max(hi[a] - lo[a], hi[b] - lo[b]);
    const uint32_t quads =
        static_cast<uint32_t>(std::clamp(std::ceil(across / std::max(cell * 64.0, 1.0e-9)), 1.0, 256.0));

    // THE RING: past the patch, out to `reach` heights from the footprint, at
    // a coarser cell -- the long shadow of a low light, which the patch's
    // edge cut off. Four strips round the patch, held to the ground.
    double outLo[3] = {lo[0], lo[1], lo[2]};
    double outHi[3] = {hi[0], hi[1], hi[2]};
    const bool ringAsked = options.reach > options.margin && options.reachLong <= 0.0;
    if (ringAsked) {
        for (const int k : {a, b}) {
            outLo[k] = objectBox.GetMin()[k] - options.reach * height;
            outHi[k] = objectBox.GetMax()[k] + options.reach * height;
            if (!groundBox.IsEmpty()) {
                outLo[k] = std::max(outLo[k], groundBox.GetMin()[k]);
                outHi[k] = std::min(outHi[k], groundBox.GetMax()[k]);
            }
        }
    }
    const double outAcross = std::max(outHi[a] - outLo[a], outHi[b] - outLo[b]);
    const bool ring = ringAsked && outAcross > across * 1.0001;
    // The conversion takes each mesh's cell as its longest side over one
    // resolution, held between the patch's cell and the ring's: so the ring's
    // is no coarser than the patch's longest side lets the patch keep its own.
    const double ringCell =
        ring ? std::max(cell, std::min(cell * std::max(options.ringCellScale, 1.0), cell * outAcross / across)) : cell;

    struct Grid {
        std::vector<std::string> points;
        std::vector<uint32_t>    indices;
    };
    const auto add = [&](Grid& grid, double a0, double a1, double b0, double b1, double step) {
        if (!(a1 - a0 > 1.0e-12) || !(b1 - b0 > 1.0e-12)) {
            return;
        }
        const auto count = [&](double length) {
            return static_cast<uint32_t>(std::clamp(std::ceil(length / std::max(step * 64.0, 1.0e-9)), 1.0, 256.0));
        };
        const uint32_t na = count(a1 - a0);
        const uint32_t nb = count(b1 - b0);
        const uint32_t first = static_cast<uint32_t>(grid.points.size());
        for (uint32_t j = 0; j <= nb; ++j) {
            for (uint32_t i = 0; i <= na; ++i) {
                double q[3] = {0, 0, 0};
                q[a] = a0 + (a1 - a0) * static_cast<double>(i) / na;
                q[b] = b0 + (b1 - b0) * static_cast<double>(j) / nb;
                q[up] = level + lift;
                std::ostringstream out;
                out.precision(9);
                out << "(" << q[0] << ", " << q[1] << ", " << q[2] << ")";
                grid.points.push_back(out.str());
            }
        }
        for (uint32_t j = 0; j < nb; ++j) {
            for (uint32_t i = 0; i < na; ++i) {
                // Counter-clockwise about the up axis for Z up; the normals
                // are authored, and the patch is two-sided either way.
                const uint32_t v = first + j * (na + 1) + i;
                grid.indices.insert(grid.indices.end(), {v, v + 1, v + 2 + na, v + 1 + na});
            }
        }
    };
    Grid patch;
    add(patch, lo[a], hi[a], lo[b], hi[b], cell);
    Grid strips;
    if (ring) {
        add(strips, outLo[a], outHi[a], outLo[b], lo[b], ringCell);
        add(strips, outLo[a], outHi[a], hi[b], outHi[b], ringCell);
        add(strips, outLo[a], lo[a], lo[b], hi[b], ringCell);
        add(strips, hi[a], outHi[a], lo[b], hi[b], ringCell);
    }
    const std::string normal = up == 2 ? "(0, 0, 1)" : "(0, 1, 0)";

    std::string binding;
    if (const UsdShadeMaterial material = UsdShadeMaterialBindingAPI(ground).ComputeBoundMaterial()) {
        binding = material.GetPath().GetString();
    }

    std::ostringstream text;
    text.precision(9);
    text << "#usda 1.0\n(\n";
    if (stage->GetRootLayer()->HasDefaultPrim()) {
        text << "    defaultPrim = \"" << stage->GetRootLayer()->GetDefaultPrim().GetString() << "\"\n";
    }
    text << "    metersPerUnit = " << UsdGeomGetStageMetersPerUnit(stage) << "\n";
    text << "    upAxis = \"" << (up == 2 ? "Z" : "Y") << "\"\n";
    text << "    subLayers = [@" << std::filesystem::absolute(source).string() << "@]\n)\n\n";
    text << "def Xform \"" << options.root << "\"\n{\n";
    const auto mesh = [&](const char* name, const Grid& grid, const double* boxLo, const double* boxHi) {
        const auto corner = [&](const double* c) {
            std::ostringstream out;
            out.precision(9);
            out << "(" << c[0] << ", " << c[1] << ", " << c[2] << ")";
            return out.str();
        };
        text << "    def Mesh \"" << name << "\" (\n        prepend apiSchemas = [\"MaterialBindingAPI\"]\n    )\n    {\n";
        text << "        uniform bool doubleSided = 1\n";
        text << "        float3[] extent = [" << corner(boxLo) << ", " << corner(boxHi) << "]\n";
        text << "        int[] faceVertexCounts = [";
        for (size_t q = 0; q < grid.indices.size() / 4; ++q) {
            text << (q ? ", " : "") << 4;
        }
        text << "]\n        int[] faceVertexIndices = [";
        for (size_t k = 0; k < grid.indices.size(); ++k) {
            text << (k ? ", " : "") << grid.indices[k];
        }
        text << "]\n        normal3f[] normals = [";
        for (size_t v = 0; v < grid.points.size(); ++v) {
            text << (v ? ", " : "") << normal;
        }
        text << "] (\n            interpolation = \"vertex\"\n        )\n";
        text << "        point3f[] points = [";
        for (size_t v = 0; v < grid.points.size(); ++v) {
            text << (v ? ", " : "") << grid.points[v];
        }
        text << "]\n";
        text << "        uniform token subdivisionScheme = \"none\"\n";
        if (!binding.empty()) {
            text << "        rel material:binding = <" << binding << ">\n";
        }
        text << "    }\n";
    };
    mesh("Patch", patch, lo, hi);
    if (ring) {
        mesh("Ring", strips, outLo, outHi);
    }
    text << "}\n";
    {
        std::ofstream out(into);
        if (!out) {
            return Error::make(ErrorCode::IoFailure, "shadow catcher: cannot write {}", into.string());
        }
        out << text.str();
    }
    ShadowCatcherStage made;
    made.stage = into;
    made.prim = ring ? "/" + options.root : "/" + options.root + "/Patch";
    made.ground = ground.GetPath().GetString();
    made.height = height;
    made.cell = cell;
    made.quads = quads;
    made.ringCell = ring ? ringCell : 0.0;
    made.ringAcross = ring ? outAcross : 0.0;
    made.min = {lo[0], lo[1], lo[2]};
    made.max = {hi[0], hi[1], hi[2]};
    log::info("{}: {} on {}, {:.3f} x {:.3f} at {:.4f} ({:.3f} high, {:.2f} heights around), "
              "{} x {} quads for a cell of {:.4g}",
              options.root, objectPath.GetString(), made.ground, hi[a] - lo[a], hi[b] - lo[b], level, height, options.margin, quads,
              quads, cell);
    if (ring) {
        log::info("{}: a ring out to {:.2f} heights, {:.3f} x {:.3f}, at a cell of {:.4g}", options.root, options.reach,
                  outHi[a] - outLo[a], outHi[b] - outLo[b], ringCell);
    }
    return made;
}

}   // namespace athenea::usd
