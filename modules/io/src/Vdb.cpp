// Copyright (c) 2026 jesus luque.

// NanoVDB 32.x's GridBuilder still names std::result_of, which C++20
// removed; libc++ keeps it behind this macro (libstdc++ keeps it as is).
#define _LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS 1
#include "athenea/io/Vdb.h"

#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

#if defined(ATHENEA_HAVE_OPENVDB)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/GridChecksum.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#endif

namespace athenea::io {

const char* vdbTypeName(VdbType type) noexcept {
    switch (type) {
        case VdbType::Float: return "float";
        case VdbType::Double: return "double";
        case VdbType::Int32: return "int";
        case VdbType::Vec3f: return "float3";
        case VdbType::Vec3d: return "double3";
        case VdbType::Mask: return "mask";
    }
    return "float";
}

Result<VdbType> vdbTypeFromDataType(const std::string& dataType) {
    if (dataType == "half" || dataType == "float") return VdbType::Float;
    if (dataType == "double") return VdbType::Double;
    if (dataType == "int") return VdbType::Int32;
    if (dataType == "half3" || dataType == "float3") return VdbType::Vec3f;
    if (dataType == "double3") return VdbType::Vec3d;
    if (dataType == "mask") return VdbType::Mask;
    return Error::make(ErrorCode::Unsupported,
                       "fieldDataType '{}' is not read (half, float, double, int, half3, float3, double3, mask are)",
                       dataType);
}

Result<VectorRole> vectorRoleFromHint(const std::string& hint) {
    if (hint.empty() || hint == "None") return VectorRole::None;
    if (hint == "Point") return VectorRole::Point;
    if (hint == "Normal") return VectorRole::Normal;
    if (hint == "Vector") return VectorRole::Vector;
    if (hint == "Color") return VectorRole::Color;
    return Error::make(ErrorCode::InvalidArgument, "vectorDataRoleHint '{}' is not one of None, Point, Normal, "
                       "Vector, Color", hint);
}

bool haveField3d() noexcept { return false; }

Result<NanoGrid> readField3dGrid(const std::filesystem::path& path, const VdbSelection&) {
    return Error::make(ErrorCode::Unsupported,
                       "'{}' is a Field3D asset, and this build reads no Field3D (the library is not in its toolchain); "
                       "convert it to OpenVDB",
                       path.string());
}

bool haveOpenVdb() noexcept {
#if defined(ATHENEA_HAVE_OPENVDB)
    return true;
#else
    return false;
#endif
}

#if defined(ATHENEA_HAVE_OPENVDB)

namespace {

void initialiseOnce() {
    static std::once_flag once;
    std::call_once(once, [] { openvdb::initialize(); });
}

}   // namespace

Result<std::vector<std::string>> vdbGridNames(const std::filesystem::path& path) {
    initialiseOnce();
    openvdb::io::File file(path.string());
    try {
        file.open(false);
        std::vector<std::string> names;
        for (auto it = file.beginName(); it != file.endName(); ++it) {
            names.push_back(it.gridName());
        }
        file.close();
        return names;
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), e.what());
    }
}

namespace {

/// A grid's value type from its header (a metadata-only grid will do).
std::optional<VdbType> typeOf(const openvdb::GridBase& grid) {
    if (grid.isType<openvdb::FloatGrid>()) return VdbType::Float;
    if (grid.isType<openvdb::DoubleGrid>()) return VdbType::Double;
    if (grid.isType<openvdb::Int32Grid>()) return VdbType::Int32;
    if (grid.isType<openvdb::Vec3SGrid>()) return VdbType::Vec3f;
    if (grid.isType<openvdb::Vec3DGrid>()) return VdbType::Vec3d;
    if (grid.isType<openvdb::MaskGrid>()) return VdbType::Mask;
    return std::nullopt;
}

}   // namespace

Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const std::string& name) {
    VdbSelection selection;
    selection.name = name;
    return readVdbGrid(path, selection);
}

Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const VdbSelection& selection) {
    initialiseOnce();
    std::optional<VdbType> declared;
    if (!selection.dataType.empty()) {
        auto parsed = vdbTypeFromDataType(selection.dataType);
        if (!parsed) return std::move(parsed).error();
        declared = *parsed;
    }
    openvdb::io::File file(path.string());
    openvdb::GridBase::Ptr base;
    try {
        file.open(false);
        // THE SELECTION, from the headers alone: the grids of the name (all
        // of them for none), in the file's order, each with its occurrence
        // among those of its name -- OpenVDB names the second "density[1]".
        // fieldIndex counts among them; fieldDataType must agree with the
        // grid chosen, and without an index it takes the first that does.
        struct Candidate {
            std::string             readName;
            std::optional<VdbType>  type;
            std::string             valueType;
        };
        std::vector<Candidate> candidates;
        std::map<std::string, int> seen;
        const openvdb::GridPtrVecPtr metas = file.readAllGridMetadata();
        for (const openvdb::GridBase::Ptr& meta : *metas) {
            const std::string gridName = meta->getName();
            const int occurrence = seen[gridName]++;
            if (!selection.name.empty() && gridName != selection.name) {
                continue;
            }
            candidates.push_back({occurrence == 0 ? gridName : gridName + "[" + std::to_string(occurrence) + "]",
                                  typeOf(*meta), meta->valueType()});
        }
        const std::string named = selection.name.empty() ? std::string("any name") : "'" + selection.name + "'";
        if (candidates.empty()) {
            return Error::make(ErrorCode::NotFound, "'{}' has no grid of {}", path.string(), named);
        }
        const Candidate* chosen = nullptr;
        if (selection.index >= 0) {
            if (static_cast<size_t>(selection.index) >= candidates.size()) {
                return Error::make(ErrorCode::NotFound, "'{}' has {} grid(s) of {}, no fieldIndex {}", path.string(),
                                   candidates.size(), named, selection.index);
            }
            chosen = &candidates[static_cast<size_t>(selection.index)];
            if (declared && chosen->type != declared) {
                return Error::make(ErrorCode::InvalidArgument, "grid {} of '{}' holds {}, not the {} its field declares",
                                   chosen->readName, path.string(), chosen->valueType, selection.dataType);
            }
        } else if (declared) {
            for (const Candidate& c : candidates) {
                if (c.type == declared) {
                    chosen = &c;
                    break;
                }
            }
            if (chosen == nullptr) {
                return Error::make(ErrorCode::InvalidArgument, "'{}' has no grid of {} holding {} (its first holds {})",
                                   path.string(), named, selection.dataType, candidates.front().valueType);
            }
        } else {
            chosen = &candidates.front();
        }
        base = file.readGrid(chosen->readName);
        file.close();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), e.what());
    }
    // The value type decides the NanoVDB grid it is laid out as, each its
    // own (a re-layout, no value converted): float, double, int32, Vec3f,
    // Vec3d and mask. A float grid stored as half was decoded by OpenVDB as
    // it was read.
    NanoGrid out;
    out.name = base->getName();
    if (!base->transform().isLinear()) {
        return Error::make(ErrorCode::Unsupported, "grid '{}': only linear transforms are read", base->getName());
    }
    if (base->isType<openvdb::FloatGrid>()) {
        out.type = VdbType::Float;
    } else if (base->isType<openvdb::DoubleGrid>()) {
        out.type = VdbType::Double;
    } else if (base->isType<openvdb::Int32Grid>()) {
        out.type = VdbType::Int32;
    } else if (base->isType<openvdb::Vec3SGrid>()) {
        out.type = VdbType::Vec3f;
    } else if (base->isType<openvdb::Vec3DGrid>()) {
        out.type = VdbType::Vec3d;
    } else if (base->isType<openvdb::MaskGrid>()) {
        out.type = VdbType::Mask;
    } else {
        return Error::make(ErrorCode::Unsupported,
                           "grid '{}' of '{}' is {}: float, double, int32, vec3s, vec3d and mask grids are read",
                           base->getName(), path.string(), base->valueType());
    }
    out.savedAsHalf = base->saveFloatAsHalf();
    // What a vector's values are, as the header says (a field's
    // vectorDataRoleHint overrides it): a direction, a normal, a position.
    switch (base->getVectorType()) {
        case openvdb::VEC_COVARIANT:
        case openvdb::VEC_COVARIANT_NORMALIZE: out.vectorRole = VectorRole::Normal; break;
        case openvdb::VEC_CONTRAVARIANT_RELATIVE: out.vectorRole = VectorRole::Vector; break;
        case openvdb::VEC_CONTRAVARIANT_ABSOLUTE: out.vectorRole = VectorRole::Point; break;
        default: out.vectorRole = VectorRole::None; break;
    }
    // The class and the background are the header's: a level set is told
    // by them, not by its values. Only a float one is made a fog.
    out.levelSet = base->getGridClass() == openvdb::GRID_LEVEL_SET;
    if (out.levelSet && out.type != VdbType::Float) {
        return Error::make(ErrorCode::Unsupported, "grid '{}': a level set of {} values; only float ones are read",
                           base->getName(), vdbTypeName(out.type));
    }
    // Tiles -- a constant region OpenVDB keeps as one value at an internal
    // node -- become leaves, so the tree the kernels walk is leaves alone:
    // the bounds, the maxima and the majorant are taken per leaf. A
    // re-layout, not a statistic. Then the re-layout, with every statistic
    // off: no bounding boxes, extrema or means are computed here, and no
    // checksum either.
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    // THE SCALAR TOPOLOGY. The medium walks float grids alone: its kernel is
    // inlined whole into the path tracer, and walking a tree whose type is
    // known only at run time indexes PNanoVDB's table of type constants
    // dynamically at every lookup, which took the Metal compiler down. So a
    // grid of any other type also gets a float grid of its topology -- the
    // same active voxels, every value 0, a copy of structure and no value
    // read -- whose values a kernel then writes from the typed grid
    // (world::VolumeSet's volumeScalarFill).
    std::shared_ptr<NanoGrid> scalar;
    const auto scalarOf = [&](const auto& typed) {
        if (out.type == VdbType::Float) {
            return;
        }
        openvdb::FloatGrid::Ptr topology = openvdb::FloatGrid::create(0.0F);
        topology->setTransform(typed.transform().copy());
        topology->tree().topologyUnion(typed.tree());
        nanovdb::GridHandle<nanovdb::HostBuffer> h =
            nanovdb::openToNanoVDB(*topology, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable);
        scalar = std::make_shared<NanoGrid>();
        scalar->words.resize((h.size() + 3) / 4, 0u);
        std::memcpy(scalar->words.data(), h.data(), h.size());
        if (const nanovdb::GridMetaData* meta = h.gridMetaData()) {
            scalar->leafCount = meta->nodeCount(0);
            scalar->lowerCount = meta->nodeCount(1);
            scalar->upperCount = meta->nodeCount(2);
            scalar->nanoType = static_cast<uint32_t>(meta->gridType());
        }
    };
    const auto lay = [&](auto typed) -> Result<void> {
        if (!typed) {
            return Error::make(ErrorCode::IoFailure, "grid '{}': not the type it names", base->getName());
        }
        typed->tree().voxelizeActiveTiles();
        handle = nanovdb::openToNanoVDB(*typed, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable);
        scalarOf(*typed);
        return ok();
    };
    try {
        Result<void> laid = ok();
        switch (out.type) {
            case VdbType::Float: laid = lay(openvdb::gridPtrCast<openvdb::FloatGrid>(base)); break;
            case VdbType::Double: laid = lay(openvdb::gridPtrCast<openvdb::DoubleGrid>(base)); break;
            case VdbType::Int32: laid = lay(openvdb::gridPtrCast<openvdb::Int32Grid>(base)); break;
            case VdbType::Vec3f: laid = lay(openvdb::gridPtrCast<openvdb::Vec3SGrid>(base)); break;
            case VdbType::Vec3d: {
                // NanoVDB 32.4's OpenVDB converter names no grid type for
                // Vec3d, so it is laid out through NanoVDB's own builder:
                // each active voxel's value copied into the same tree shape
                // (inactive voxels are the background), a copy as the
                // converter's.
                auto typed = openvdb::gridPtrCast<openvdb::Vec3DGrid>(base);
                if (!typed) return Error::make(ErrorCode::IoFailure, "grid '{}': not a vec3d grid", base->getName());
                typed->tree().voxelizeActiveTiles();
                const openvdb::Vec3d bg = typed->background();
                nanovdb::GridBuilder<nanovdb::Vec3d> builder(nanovdb::Vec3d(bg[0], bg[1], bg[2]));
                builder.setStats(nanovdb::StatsMode::Disable);
                builder.setChecksum(nanovdb::ChecksumMode::Disable);
                auto acc = builder.getAccessor();
                for (auto it = typed->tree().cbeginValueOn(); it; ++it) {
                    const openvdb::Coord c = it.getCoord();
                    const openvdb::Vec3d v = *it;
                    acc.setValue(nanovdb::Coord(c.x(), c.y(), c.z()), nanovdb::Vec3d(v[0], v[1], v[2]));
                }
                handle = builder.getHandle(1.0, nanovdb::Vec3d(0.0), base->getName());
                scalarOf(*typed);
                break;
            }
            case VdbType::Mask: laid = lay(openvdb::gridPtrCast<openvdb::MaskGrid>(base)); break;
        }
        if (!laid) return std::move(laid).error();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "grid '{}': cannot lay out as NanoVDB: {}", base->getName(),
                           e.what());
    }
    out.words.resize((handle.size() + 3) / 4, 0u);
    std::memcpy(out.words.data(), handle.data(), handle.size());
    // The header's counts and type: bookkeeping the kernels index by.
    if (const nanovdb::GridMetaData* meta = handle.gridMetaData()) {
        out.leafCount = meta->nodeCount(0);
        out.lowerCount = meta->nodeCount(1);
        out.upperCount = meta->nodeCount(2);
        out.nanoType = static_cast<uint32_t>(meta->gridType());
    }
    out.voxelSize = base->transform().voxelSize()[0];
    if (scalar) {
        scalar->name = out.name;
        scalar->voxelSize = out.voxelSize;
        out.scalar = std::move(scalar);
    }
    if (out.levelSet) {
        out.background = openvdb::gridPtrCast<openvdb::FloatGrid>(base)->background();
    }
    const openvdb::math::Mat4d m = base->transform().baseMap()->getAffineMap()->getMat4();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.indexToWorld[static_cast<size_t>(r * 4 + c)] = m(r, c);
        }
    }
    return out;
}

Result<NanoGrid> readVdbVec3Grid(const std::filesystem::path& path, const std::string& name) {
    initialiseOnce();
    openvdb::io::File file(path.string());
    openvdb::GridBase::Ptr base;
    try {
        file.open(false);
        if (!file.hasGrid(name)) {
            return Error::make(ErrorCode::NotFound, "'{}' has no grid '{}'", path.string(), name);
        }
        base = file.readGrid(name);
        file.close();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot read '{}': {}", path.string(), e.what());
    }
    auto grid = openvdb::gridPtrCast<openvdb::Vec3SGrid>(base);
    if (!grid) {
        return Error::make(ErrorCode::Unsupported, "grid '{}' of '{}' is {}, not vec3s", base->getName(),
                           path.string(), base->valueType());
    }
    if (!grid->transform().isLinear()) {
        return Error::make(ErrorCode::Unsupported, "grid '{}': only linear transforms are read", base->getName());
    }
    grid->tree().voxelizeActiveTiles();
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    try {
        handle = nanovdb::openToNanoVDB(*grid, nanovdb::StatsMode::Disable, nanovdb::ChecksumMode::Disable);
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "grid '{}': cannot lay out as NanoVDB: {}", base->getName(),
                           e.what());
    }
    NanoGrid out;
    out.name = base->getName();
    out.components = 3;
    out.words.resize((handle.size() + 3) / 4, 0u);
    std::memcpy(out.words.data(), handle.data(), handle.size());
    out.leafCount = handle.gridMetaData() != nullptr ? handle.gridMetaData()->nodeCount(0) : 0;
    out.voxelSize = grid->transform().voxelSize()[0];
    const openvdb::math::Mat4d m = grid->transform().baseMap()->getAffineMap()->getMat4();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.indexToWorld[static_cast<size_t>(r * 4 + c)] = m(r, c);
        }
    }
    return out;
}

Result<void> writeNanoGrids(const std::filesystem::path& path, std::span<const NanoGrid> grids) {
    initialiseOnce();
    try {
        openvdb::GridPtrVec out;
        for (const NanoGrid& grid : grids) {
            if (grid.components != 1) {
                return Error::make(ErrorCode::Unsupported, "'{}': only float grids are written", grid.name);
            }
            // NanoVDB wants its buffer 32-byte aligned, which a vector of
            // words does not promise: the words are copied into one that is.
            auto buffer = nanovdb::HostBuffer::create(grid.words.size() * 4);
            std::memcpy(buffer.data(), grid.words.data(), grid.words.size() * 4);
            nanovdb::GridHandle<nanovdb::HostBuffer> handle(std::move(buffer));
            const nanovdb::NanoGrid<float>* nano = handle.grid<float>();
            if (nano == nullptr) {
                return Error::make(ErrorCode::InvalidArgument, "'{}': not a NanoVDB float grid", grid.name);
            }
            openvdb::FloatGrid::Ptr back = nanovdb::nanoToOpenVDB(*nano);
            back->setName(grid.name);
            back->setGridClass(openvdb::GRID_FOG_VOLUME);
            out.push_back(back);
        }
        openvdb::io::File file(path.string());
        file.write(out);
        file.close();
        return ok();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}': {}", path.string(), e.what());
    }
}

namespace {

openvdb::GridBase::Ptr boxGrid(const VdbGridSpec& spec) {
    const auto make = [&](auto grid, auto valueOf) -> openvdb::GridBase::Ptr {
        grid->setName(spec.name);
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        grid->setTransform(openvdb::math::Transform::createLinearTransform(spec.voxelSize));
        for (const VdbBox& box : spec.boxes) {
            const openvdb::CoordBBox bbox(openvdb::Coord(box.min[0], box.min[1], box.min[2]),
                                          openvdb::Coord(box.max[0] - 1, box.max[1] - 1, box.max[2] - 1));
            grid->fill(bbox, valueOf(box), true);
        }
        return grid;
    };
    openvdb::GridBase::Ptr grid;
    switch (spec.type) {
        case VdbType::Float:
            grid = make(openvdb::FloatGrid::create(0.0F), [](const VdbBox& b) { return float(b.value); });
            break;
        case VdbType::Double:
            grid = make(openvdb::DoubleGrid::create(0.0), [](const VdbBox& b) { return double(b.value); });
            break;
        case VdbType::Int32:
            grid = make(openvdb::Int32Grid::create(0), [](const VdbBox& b) { return int32_t(b.value); });
            break;
        case VdbType::Vec3f:
            grid = make(openvdb::Vec3SGrid::create(openvdb::Vec3s(0.0F)), [](const VdbBox& b) {
                return openvdb::Vec3s(float(b.vector[0]), float(b.vector[1]), float(b.vector[2]));
            });
            break;
        case VdbType::Vec3d:
            grid = make(openvdb::Vec3DGrid::create(openvdb::Vec3d(0.0)), [](const VdbBox& b) {
                return openvdb::Vec3d(b.vector[0], b.vector[1], b.vector[2]);
            });
            break;
        case VdbType::Mask:
            grid = make(openvdb::MaskGrid::create(false), [](const VdbBox&) { return true; });
            break;
    }
    grid->setSaveFloatAsHalf(spec.saveAsHalf);
    switch (spec.vectorRole) {
        case VectorRole::Normal: grid->setVectorType(openvdb::VEC_COVARIANT); break;
        case VectorRole::Vector: grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE); break;
        case VectorRole::Point: grid->setVectorType(openvdb::VEC_CONTRAVARIANT_ABSOLUTE); break;
        default: break;
    }
    return grid;
}

}   // namespace

Result<void> writeVdbGrids(const std::filesystem::path& path, std::span<const VdbGridSpec> grids) {
    initialiseOnce();
    try {
        openvdb::GridPtrVec out;
        for (const VdbGridSpec& spec : grids) {
            out.push_back(boxGrid(spec));
        }
        openvdb::io::File file(path.string());
        file.write(out);
        file.close();
        return ok();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}': {}", path.string(), e.what());
    }
}

Result<void> writeVdbBoxes(const std::filesystem::path& path, const std::string& name, double voxelSize,
                           std::span<const VdbBox> boxes, VdbType type, bool saveAsHalf) {
    VdbGridSpec spec;
    spec.name = name;
    spec.type = type;
    spec.voxelSize = voxelSize;
    spec.boxes.assign(boxes.begin(), boxes.end());
    spec.saveAsHalf = saveAsHalf;
    return writeVdbGrids(path, std::span<const VdbGridSpec>(&spec, 1));
}

Result<void> writeVdbLevelSetBox(const std::filesystem::path& path, const std::string& name, double voxelSize,
                                 const VdbBox& box, float halfWidth) {
    initialiseOnce();
    try {
        const openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(voxelSize);
        const openvdb::math::BBox<openvdb::Vec3s> bbox(
            openvdb::Vec3s(float(box.min[0] * voxelSize), float(box.min[1] * voxelSize), float(box.min[2] * voxelSize)),
            openvdb::Vec3s(float(box.max[0] * voxelSize), float(box.max[1] * voxelSize), float(box.max[2] * voxelSize)));
        openvdb::FloatGrid::Ptr grid = openvdb::tools::createLevelSetBox<openvdb::FloatGrid>(bbox, *xform, halfWidth);
        grid->setName(name);
        grid->setGridClass(openvdb::GRID_LEVEL_SET);
        openvdb::io::File file(path.string());
        openvdb::GridPtrVec grids{grid};
        file.write(grids);
        file.close();
        return ok();
    } catch (const std::exception& e) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}': {}", path.string(), e.what());
    }
}

#else

Result<std::vector<std::string>> vdbGridNames(const std::filesystem::path& path) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const std::string&) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<NanoGrid> readVdbVec3Grid(const std::filesystem::path& path, const std::string&) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<void> writeNanoGrids(const std::filesystem::path& path, std::span<const NanoGrid>) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const VdbSelection&) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<void> writeVdbBoxes(const std::filesystem::path& path, const std::string&, double, std::span<const VdbBox>,
                           VdbType, bool) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<void> writeVdbGrids(const std::filesystem::path& path, std::span<const VdbGridSpec>) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

Result<void> writeVdbLevelSetBox(const std::filesystem::path& path, const std::string&, double, const VdbBox&, float) {
    return Error::make(ErrorCode::Unsupported, "'{}': built without OpenVDB", path.string());
}

#endif

}   // namespace athenea::io
