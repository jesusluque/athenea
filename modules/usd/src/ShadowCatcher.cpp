// Copyright (c) 2026 jesus luque.
#include "athenea/usd/ShadowCatcher.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

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
    const double reach = options.margin * height;
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};
    for (const int k : {a, b}) {
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
    const double cell = options.cell > 0.0 ? options.cell : height / 100.0;
    const double across = std::max(hi[a] - lo[a], hi[b] - lo[b]);
    const uint32_t quads =
        static_cast<uint32_t>(std::clamp(std::ceil(across / std::max(cell * 64.0, 1.0e-9)), 1.0, 256.0));
    const uint32_t side = quads + 1;
    const auto point = [&](uint32_t i, uint32_t j) {
        double p[3] = {0, 0, 0};
        p[a] = lo[a] + (hi[a] - lo[a]) * static_cast<double>(i) / quads;
        p[b] = lo[b] + (hi[b] - lo[b]) * static_cast<double>(j) / quads;
        p[up] = level + lift;
        std::ostringstream out;
        out.precision(9);
        out << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
        return out.str();
    };
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
    text << "def Xform \"AtheneaShadowCatcher\"\n{\n";
    text << "    def Mesh \"Patch\" (\n        prepend apiSchemas = [\"MaterialBindingAPI\"]\n    )\n    {\n";
    text << "        uniform bool doubleSided = 1\n";
    text << "        float3[] extent = [" << point(0, 0) << ", " << point(quads, quads) << "]\n";
    text << "        int[] faceVertexCounts = [";
    for (uint32_t q = 0; q < quads * quads; ++q) {
        text << (q ? ", " : "") << 4;
    }
    text << "]\n        int[] faceVertexIndices = [";
    for (uint32_t j = 0; j < quads; ++j) {
        for (uint32_t i = 0; i < quads; ++i) {
            const uint32_t v = j * side + i;
            // Counter-clockwise about the up axis for Z up; the normals are
            // authored, and the patch is two-sided either way.
            text << ((i || j) ? ", " : "") << v << ", " << v + 1 << ", " << v + 1 + side << ", " << v + side;
        }
    }
    text << "]\n        normal3f[] normals = [";
    for (uint32_t v = 0; v < side * side; ++v) {
        text << (v ? ", " : "") << normal;
    }
    text << "] (\n            interpolation = \"vertex\"\n        )\n";
    text << "        point3f[] points = [";
    for (uint32_t j = 0; j < side; ++j) {
        for (uint32_t i = 0; i < side; ++i) {
            text << ((i || j) ? ", " : "") << point(i, j);
        }
    }
    text << "]\n";
    text << "        uniform token subdivisionScheme = \"none\"\n";
    if (!binding.empty()) {
        text << "        rel material:binding = <" << binding << ">\n";
    }
    text << "    }\n}\n";
    {
        std::ofstream out(into);
        if (!out) {
            return Error::make(ErrorCode::IoFailure, "shadow catcher: cannot write {}", into.string());
        }
        out << text.str();
    }
    ShadowCatcherStage made;
    made.stage = into;
    made.prim = "/AtheneaShadowCatcher/Patch";
    made.ground = ground.GetPath().GetString();
    made.height = height;
    made.cell = cell;
    made.quads = quads;
    made.min = {lo[0], lo[1], lo[2]};
    made.max = {hi[0], hi[1], hi[2]};
    log::info("shadow catcher: {} on {}, {:.3f} x {:.3f} at {:.4f} ({:.3f} high, {:.2f} heights around), "
              "{} x {} quads for a cell of {:.4g}",
              objectPath.GetString(), made.ground, hi[a] - lo[a], hi[b] - lo[b], level, height, options.margin, quads,
              quads, cell);
    return made;
}

}   // namespace athenea::usd
