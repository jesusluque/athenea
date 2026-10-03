// Copyright (c) 2026 jesus luque.
//
// A layer is migrated as layers are: opened, its content moved into an
// anonymous layer (so the original, cached in USD's registry, is never
// edited), every spec walked -- variants too -- and the copy exported under
// the new name. Nothing is composed: each layer's opinions stay its own.
#include "athenea/usd/Migrate.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <optional>
#include <set>

#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/propertySpec.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/relationshipSpec.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/sdf/variantSetSpec.h>
#include <pxr/usd/sdf/variantSpec.h>
#include <pxr/usd/sdf/zipFile.h>
#include <pxr/usd/usd/tokens.h>

#include "athenea/lod/Athc.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace athenea::usd {
namespace {

namespace fs = std::filesystem;

std::string extensionOf(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext;
}
bool isLayerFile(const fs::path& p) {
    const std::string e = extensionOf(p);
    return e == ".usda" || e == ".usdc" || e == ".usd";
}
bool isPackage(const fs::path& p) {
    return extensionOf(p) == ".usdz";
}
bool isCloudFile(const fs::path& p) {
    const std::string e = extensionOf(p);
    return e == ".lrtc" || e == ".athc";
}
bool endsWithLrtc(const std::string& s) {
    if (s.size() < 5) {
        return false;
    }
    std::string tail = s.substr(s.size() - 5);
    std::transform(tail.begin(), tail.end(), tail.begin(), [](unsigned char c) { return std::tolower(c); });
    return tail == ".lrtc";
}
std::string asAthc(const std::string& s) {
    return endsWithLrtc(s) ? s.substr(0, s.size() - 5) + ".athc" : s;
}

/// Every `lrt` component of a namespaced name becomes `athenea`:
/// primvars:lrt:splat:ior, lrt:pathTotal, outputs:lrt:surface.
std::string renamedName(const std::string& name) {
    std::string out;
    size_t start = 0;
    while (true) {
        const size_t colon = name.find(':', start);
        const std::string part = name.substr(start, colon == std::string::npos ? std::string::npos : colon - start);
        out += part == "lrt" ? "athenea" : part;
        if (colon == std::string::npos) {
            break;
        }
        out += ':';
        start = colon + 1;
    }
    return out;
}

/// LrtSplatEditAPI -> AtheneaSplatEditAPI; an instance name after a colon
/// (a multiple-apply schema) stays.
std::optional<std::string> renamedSchema(const std::string& s) {
    if (s.size() > 3 && s.compare(0, 3, "Lrt") == 0 && std::isupper(static_cast<unsigned char>(s[3]))) {
        return "Athenea" + s.substr(3);
    }
    return std::nullopt;
}

std::optional<std::string> renamedRenderer(const std::string& s) {
    if (s == "lrt") return std::string("athenea");
    if (s == "HdLrtRendererPlugin") return std::string("HdAtheneaRendererPlugin");
    if (s == "hdLrt") return std::string("hdAthenea");
    return std::nullopt;
}

SdfPath renamedPath(const SdfPath& path) {
    if (!path.IsPropertyPath()) {
        return path;
    }
    const std::string name = path.GetName();
    const std::string to = renamedName(name);
    return to == name ? path : path.ReplaceName(TfToken(to));
}

bool under(const fs::path& path, const fs::path& root) {
    const fs::path rel = path.lexically_relative(root);
    return !rel.empty() && *rel.begin() != "..";
}

fs::path resolvedPath(const fs::path& path) {
    std::error_code ec;
    fs::path c = fs::weakly_canonical(fs::absolute(path, ec), ec);
    return ec ? fs::absolute(path, ec).lexically_normal() : c;
}

struct LayerJob {
    fs::path    src, dst;
    fs::path    srcDir, dstDir;
    std::string label;
    bool        anchor = true;    // relative paths to files not copied are made absolute
    bool        follow = false;   // --recursive
};

class Migrator {
public:
    explicit Migrator(MigrateOptions options) : options_(std::move(options)) {}

    Result<MigrateReport> run(const fs::path& in, const fs::path& out) {
        const fs::path src = resolvedPath(in);
        const fs::path dst = resolvedPath(out);
        std::error_code ec;
        if (!fs::exists(src, ec)) {
            return Error(ErrorCode::NotFound, in.string() + ": no such file");
        }
        if (src == dst || fs::equivalent(src, dst, ec)) {
            return Error(ErrorCode::InvalidArgument,
                         out.string() + " is the input; a migration writes a copy and never overwrites what it reads");
        }
        const bool layer = isLayerFile(src), package = isPackage(src), cloud = isCloudFile(src);
        if (!layer && !package && !cloud) {
            return Error(ErrorCode::InvalidArgument,
                         in.string() + ": migrate reads .usda, .usdc, .usd, .usdz and .lrtc files");
        }
        if ((layer && !isLayerFile(dst)) || (package && !isPackage(dst)) || (cloud && extensionOf(dst) != ".athc")) {
            return Error(ErrorCode::InvalidArgument,
                         out.string() + ": a " + extensionOf(src) + " migrates to " +
                             (layer ? ".usda, .usdc or .usd" : package ? ".usdz" : ".athc"));
        }
        root_ = resolvedPath(options_.root.empty() ? src.parent_path() : options_.root);
        outRoot_ = dst.parent_path();
        if (options_.recursive && !under(src, root_)) {
            return Error(ErrorCode::InvalidArgument, "--root " + root_.string() + " does not hold " + src.string());
        }
        if (options_.recursive && outRoot_ == root_) {
            return Error(ErrorCode::InvalidArgument,
                         "with --recursive the copies are written beside -o as they lie beside the input; " +
                             outRoot_.string() + " is where the originals are");
        }
        ATHENEA_TRY(node(src, dst));
        return std::move(report_);
    }

private:
    MigrateOptions        options_;
    MigrateReport         report_;
    fs::path              root_, outRoot_;
    std::map<fs::path, fs::path> done_;
    std::optional<Error>  failure_;

    void note(MigrateChange::Kind kind, const std::string& file, const std::string& where, const std::string& from,
              const std::string& to = {}) {
        report_.changes.push_back({kind, file, where, from, to});
    }

    fs::path mirror(const fs::path& src) const {
        fs::path m = outRoot_ / src.lexically_relative(root_);
        if (extensionOf(m) == ".lrtc") {
            m.replace_extension(".athc");
        }
        return m;
    }

    /// One file and, with --recursive, what it names. Returns where it went.
    Result<fs::path> node(const fs::path& src, const fs::path& dst) {
        if (auto it = done_.find(src); it != done_.end()) {
            return it->second;   // already written, or being written (a cycle)
        }
        if (resolvedPath(dst) == src) {
            return Error(ErrorCode::InvalidArgument, dst.string() + " would overwrite the original it migrates");
        }
        done_[src] = dst;
        std::error_code ec;
        fs::create_directories(dst.parent_path(), ec);
        if (ec) {
            return Error(ErrorCode::IoFailure, "cannot create " + dst.parent_path().string() + ": " + ec.message());
        }
        if (isCloudFile(src)) {
            auto converted = lod::migrateLrtc(src, dst);
            if (!converted) return std::move(converted).error();
            if (*converted) {
                note(MigrateChange::Kind::Asset, src.string(), "", "LRTC version 1", "ATHC version 2");
            } else {
                note(MigrateChange::Kind::Warning, src.string(), "", "already a .athc: copied as it is");
            }
        } else if (isPackage(src)) {
            ATHENEA_TRY(package(src, dst));
        } else {
            LayerJob job;
            job.src = src;
            job.dst = dst;
            job.srcDir = src.parent_path();
            job.dstDir = resolvedPath(dst.parent_path());
            job.label = src.string();
            job.follow = options_.recursive;
            ATHENEA_TRY(layer(job));
        }
        report_.written.push_back(dst);
        note(MigrateChange::Kind::File, src.string(), "", src.string(), dst.string());
        return dst;
    }

    // --- asset paths ----------------------------------------------------------

    std::string relativeFrom(const fs::path& target, const fs::path& dir, const std::string& original) const {
        std::string r = target.lexically_relative(dir).generic_string();
        if (r.empty()) {
            return target.generic_string();
        }
        if (r.compare(0, 1, ".") != 0) {
            r = "./" + r;
        }
        // The same path written as the original wrote it, where it is the same.
        if (fs::path(r).lexically_normal() == fs::path(original).lexically_normal()) {
            return original;
        }
        return r;
    }

    std::string asset(const LayerJob& job, const std::string& path, const std::string& where) {
        if (path.empty()) {
            return path;
        }
        // Expressions and package-relative paths are not followed.
        if (path.front() == '`' || path.find('[') != std::string::npos) {
            if (endsWithLrtc(path) || path.find(".lrtc") != std::string::npos) {
                note(MigrateChange::Kind::Warning, job.label, where, path,
                     "names a .lrtc inside an expression or a package path: rewrite it by hand");
            }
            return path;
        }
        const fs::path written(path);
        const bool absolute = written.is_absolute();
        const fs::path resolved = resolvedPath(absolute ? written : job.srcDir / written);
        std::error_code ec;
        const bool exists = fs::exists(resolved, ec);
        const bool followable = isLayerFile(resolved) || isPackage(resolved) || isCloudFile(resolved);
        if (job.follow && exists && followable && under(resolved, root_)) {
            auto dst = node(resolved, mirror(resolved));
            if (!dst) {
                if (!failure_) failure_ = std::move(dst).error();
                return path;
            }
            const std::string to = absolute ? dst->generic_string() : relativeFrom(*dst, job.dstDir, path);
            if (to != path) {
                note(MigrateChange::Kind::Asset, job.label, where, path, to);
            }
            return to;
        }
        std::string to = asAthc(path);
        if (to != path) {
            note(MigrateChange::Kind::Asset, job.label, where, path, to);
        }
        if (job.anchor && !absolute && job.srcDir != job.dstDir &&
            (exists || path.find("<UDIM>") != std::string::npos)) {
            const std::string anchored = asAthc(resolved.generic_string());
            note(MigrateChange::Kind::Anchored, job.label, where, to, anchored);
            to = anchored;
        }
        if (endsWithLrtc(path) && job.anchor) {
            const fs::path athc = resolvedPath(fs::path(to).is_absolute() ? fs::path(to) : job.dstDir / to);
            if (!fs::exists(athc, ec)) {
                note(MigrateChange::Kind::Warning, job.label, where, to,
                     "no such file yet: `athenea migrate " + resolved.string() + " -o " + athc.string() + "`" +
                         (job.follow ? " (it lies outside --root)" : " (or --recursive)"));
            }
        }
        return to;
    }

    /// An asset or asset array value, rewritten where it names one.
    bool assetValue(const LayerJob& job, VtValue& value, const std::string& where) {
        if (value.IsHolding<SdfAssetPath>()) {
            const SdfAssetPath& a = value.UncheckedGet<SdfAssetPath>();
            const std::string to = asset(job, a.GetAssetPath(), where);
            if (to != a.GetAssetPath()) {
                value = VtValue(SdfAssetPath(to));
                return true;
            }
        } else if (value.IsHolding<VtArray<SdfAssetPath>>()) {
            VtArray<SdfAssetPath> array = value.UncheckedGet<VtArray<SdfAssetPath>>();
            bool changed = false;
            for (SdfAssetPath& a : array) {
                const std::string to = asset(job, a.GetAssetPath(), where);
                if (to != a.GetAssetPath()) {
                    a = SdfAssetPath(to);
                    changed = true;
                }
            }
            if (changed) {
                value = VtValue(array);
                return true;
            }
        }
        return false;
    }

    bool dictionary(const LayerJob& job, VtDictionary& dict, const std::string& where) {
        bool changed = false;
        VtDictionary out;
        for (const auto& [key, held] : dict) {
            VtValue value = held;
            if (value.IsHolding<VtDictionary>()) {
                VtDictionary inner = value.UncheckedGet<VtDictionary>();
                if (dictionary(job, inner, where)) {
                    value = VtValue(inner);
                    changed = true;
                }
            } else if (assetValue(job, value, where)) {
                changed = true;
            } else if (value.IsHolding<std::string>() && endsWithLrtc(value.UncheckedGet<std::string>())) {
                note(MigrateChange::Kind::Warning, job.label, where, value.UncheckedGet<std::string>(),
                     "a string naming a .lrtc (a clip template?): left as it is");
            }
            const std::string to = renamedName(key);
            if (to != key) {
                note(MigrateChange::Kind::Metadata, job.label, where, key, to);
                changed = true;
            }
            out[to] = value;
        }
        if (changed) {
            dict = std::move(out);
        }
        return changed;
    }

    void dictionaryField(const LayerJob& job, const SdfSpecHandle& spec, const TfToken& field) {
        if (!spec->HasField(field)) {
            return;
        }
        VtValue value = spec->GetField(field);
        if (!value.IsHolding<VtDictionary>()) {
            return;
        }
        VtDictionary dict = value.UncheckedGet<VtDictionary>();
        if (dictionary(job, dict, spec->GetPath().GetString())) {
            spec->SetField(field, VtValue(dict));
        }
    }

    template <class Op, class Edit>
    void listOpField(const SdfSpecHandle& spec, const TfToken& field, Edit edit) {
        if (!spec->HasField(field)) {
            return;
        }
        VtValue value = spec->GetField(field);
        if (!value.IsHolding<Op>()) {
            return;
        }
        Op op = value.UncheckedGet<Op>();
        if (op.ModifyOperations(edit)) {
            spec->SetField(field, VtValue(op));
        }
    }

    // --- specs ---------------------------------------------------------------

    void paths(const LayerJob& job, const SdfSpecHandle& spec, const TfToken& field) {
        const std::string where = spec->GetPath().GetString();
        listOpField<SdfPathListOp>(spec, field, [&](const SdfPath& p) -> std::optional<SdfPath> {
            const SdfPath to = renamedPath(p);
            if (to != p) {
                note(MigrateChange::Kind::Target, job.label, where, p.GetString(), to.GetString());
            }
            return to;
        });
    }

    void property(const LayerJob& job, const SdfPrimSpecHandle& prim, const SdfPropertySpecHandle& prop) {
        const std::string where = prop->GetPath().GetString();
        if (SdfAttributeSpecHandle attr = TfDynamic_cast<SdfAttributeSpecHandle>(prop)) {
            const SdfValueTypeName type = attr->GetTypeName();
            const bool assets = type == SdfValueTypeNames->Asset || type == SdfValueTypeNames->AssetArray;
            const bool renderer = prop->GetName() == "hydra:rendererName";
            if (assets || renderer) {
                const auto fix = [&](VtValue& value) {
                    if (assets) {
                        return assetValue(job, value, where);
                    }
                    std::optional<std::string> to;
                    std::string from;
                    if (value.IsHolding<TfToken>()) {
                        from = value.UncheckedGet<TfToken>().GetString();
                        to = renamedRenderer(from);
                        if (to) value = VtValue(TfToken(*to));
                    } else if (value.IsHolding<std::string>()) {
                        from = value.UncheckedGet<std::string>();
                        to = renamedRenderer(from);
                        if (to) value = VtValue(*to);
                    }
                    if (to) note(MigrateChange::Kind::Value, job.label, where, from, *to);
                    return to.has_value();
                };
                if (attr->HasField(SdfFieldKeys->Default)) {
                    VtValue value = attr->GetField(SdfFieldKeys->Default);
                    if (fix(value)) {
                        attr->SetField(SdfFieldKeys->Default, value);
                    }
                }
                if (attr->HasField(SdfFieldKeys->TimeSamples)) {
                    VtValue held = attr->GetField(SdfFieldKeys->TimeSamples);
                    if (held.IsHolding<SdfTimeSampleMap>()) {
                        SdfTimeSampleMap samples = held.UncheckedGet<SdfTimeSampleMap>();
                        bool changed = false;
                        for (auto& [time, value] : samples) {
                            changed = fix(value) || changed;
                        }
                        if (changed) {
                            attr->SetField(SdfFieldKeys->TimeSamples, VtValue(samples));
                        }
                    }
                }
            }
            paths(job, prop, SdfFieldKeys->ConnectionPaths);
        } else {
            paths(job, prop, SdfFieldKeys->TargetPaths);
        }
        dictionaryField(job, prop, SdfFieldKeys->CustomData);

        const std::string name = prop->GetName();
        const std::string to = renamedName(name);
        if (to == name) {
            return;
        }
        const SdfPath target = prim->GetPath().AppendProperty(TfToken(to));
        if (prim->GetLayer()->GetPropertyAtPath(target)) {
            note(MigrateChange::Kind::Warning, job.label, where,
                 name, "not renamed: " + to + " is authored beside it, and wins");
            return;
        }
        std::string whyNot;
        if (!prop->CanSetName(to, &whyNot) || !prop->SetName(to)) {
            note(MigrateChange::Kind::Warning, job.label, where, name, "not renamed: " + whyNot);
            return;
        }
        note(MigrateChange::Kind::Property, job.label, prim->GetPath().GetString(), name, to);
    }

    void prim(const LayerJob& job, const SdfPrimSpecHandle& spec) {
        const std::string where = spec->GetPath().GetString();
        if (spec->GetSpecType() != SdfSpecTypePseudoRoot) {
            listOpField<SdfTokenListOp>(spec, UsdTokens->apiSchemas, [&](const TfToken& t) -> std::optional<TfToken> {
                if (auto to = renamedSchema(t.GetString())) {
                    note(MigrateChange::Kind::Schema, job.label, where, t.GetString(), *to);
                    return TfToken(*to);
                }
                return t;
            });
            listOpField<SdfReferenceListOp>(
                spec, SdfFieldKeys->References, [&](const SdfReference& r) -> std::optional<SdfReference> {
                    SdfReference out = r;
                    out.SetAssetPath(asset(job, r.GetAssetPath(), where));
                    return out;
                });
            listOpField<SdfPayloadListOp>(spec, SdfFieldKeys->Payload,
                                          [&](const SdfPayload& p) -> std::optional<SdfPayload> {
                                              SdfPayload out = p;
                                              out.SetAssetPath(asset(job, p.GetAssetPath(), where));
                                              return out;
                                          });
            dictionaryField(job, spec, UsdTokens->clips);
            dictionaryField(job, spec, SdfFieldKeys->CustomData);
            if (spec->HasField(SdfFieldKeys->PropertyOrder)) {
                VtValue held = spec->GetField(SdfFieldKeys->PropertyOrder);
                if (held.IsHolding<TfTokenVector>()) {
                    TfTokenVector order = held.UncheckedGet<TfTokenVector>();
                    bool changed = false;
                    for (TfToken& t : order) {
                        const std::string to = renamedName(t.GetString());
                        if (to != t.GetString()) {
                            t = TfToken(to);
                            changed = true;
                        }
                    }
                    if (changed) {
                        spec->SetField(SdfFieldKeys->PropertyOrder, VtValue(order));
                    }
                }
            }
            std::vector<SdfPropertySpecHandle> props;
            for (const SdfPropertySpecHandle& p : spec->GetProperties()) {
                props.push_back(p);
            }
            for (const SdfPropertySpecHandle& p : props) {
                property(job, spec, p);
            }
            const SdfVariantSetSpecHandleVector sets = spec->GetVariantSets();
            for (const SdfVariantSetSpecHandle& set : sets) {
                for (const SdfVariantSpecHandle& variant : set->GetVariantList()) {
                    prim(job, variant->GetPrimSpec());
                }
            }
        }
        std::vector<SdfPrimSpecHandle> children;
        for (const SdfPrimSpecHandle& child : spec->GetNameChildren()) {
            children.push_back(child);
        }
        for (const SdfPrimSpecHandle& child : children) {
            prim(job, child);
        }
    }

    Result<void> layer(const LayerJob& job) {
        SdfLayerRefPtr source = SdfLayer::FindOrOpen(job.src.string());
        if (!source) {
            return Error(ErrorCode::IoFailure, job.src.string() + ": USD could not open it as a layer");
        }
        // A copy: the opened layer stays as the file says, for whoever opens it next.
        SdfLayerRefPtr work = SdfLayer::CreateAnonymous("migrate");
        work->TransferContent(source);

        std::vector<std::string> subLayers;
        for (const std::string& s : work->GetSubLayerPaths()) {
            subLayers.push_back(s);
        }
        const std::vector<SdfLayerOffset> offsets = work->GetSubLayerOffsets();
        bool subChanged = false;
        for (std::string& s : subLayers) {
            const std::string to = asset(job, s, "subLayers");
            subChanged = subChanged || to != s;
            s = to;
        }
        if (subChanged) {
            work->SetSubLayerPaths(subLayers);
            for (size_t i = 0; i < offsets.size() && i < subLayers.size(); ++i) {
                work->SetSubLayerOffset(offsets[i], static_cast<int>(i));
            }
        }
        VtDictionary layerData = work->GetCustomLayerData();
        if (dictionary(job, layerData, "customLayerData")) {
            work->SetCustomLayerData(layerData);
        }
        prim(job, work->GetPseudoRoot());
        if (failure_) {
            return *failure_;
        }

        SdfFileFormat::FileFormatArguments args;
        if (extensionOf(job.dst) == ".usd") {
            // .usd is either encoding: keep the one the input had.
            args["format"] = isLayerFile(job.src) && extensionOf(job.src) != ".usd"
                                 ? extensionOf(job.src).substr(1)
                                 : source->GetFileFormat()->GetFormatId().GetString();
        }
        if (!work->Export(job.dst.string(), std::string(), args)) {
            return Error(ErrorCode::IoFailure, "cannot write " + job.dst.string());
        }
        return ok();
    }

    /// A .usdz: its files extracted beside the output, each layer migrated as
    /// itself (everything in a package names everything else relatively), a
    /// .lrtc converted under its new name, and the package written again in
    /// the same order -- the first file is still the root layer.
    Result<void> package(const fs::path& src, const fs::path& dst) {
        SdfZipFile zip = SdfZipFile::Open(src.string());
        if (!zip) {
            return Error(ErrorCode::IoFailure, src.string() + ": not a readable .usdz");
        }
        const fs::path work = dst.parent_path() / ("." + dst.filename().string() + ".migrating");
        std::error_code ec;
        fs::remove_all(work, ec);
        std::vector<std::pair<std::string, fs::path>> entries;
        for (auto it = zip.begin(); it != zip.end(); ++it) {
            const std::string name = *it;
            const SdfZipFile::FileInfo info = it.GetFileInfo();
            if (info.compressionMethod != 0 || info.encrypted) {
                fs::remove_all(work, ec);
                return Error(ErrorCode::Unsupported, src.string() + ": " + name + " is compressed or encrypted");
            }
            const fs::path extracted = work / "in" / name;
            fs::create_directories(extracted.parent_path(), ec);
            std::ofstream file(extracted, std::ios::binary | std::ios::trunc);
            file.write(it.GetFile(), static_cast<std::streamsize>(info.size));
            file.close();
            if (!file) {
                fs::remove_all(work, ec);
                return Error(ErrorCode::IoFailure, "cannot extract " + name + " from " + src.string());
            }
            entries.emplace_back(name, extracted);
        }
        const auto failed = [&](Error error) -> Result<void> {
            fs::remove_all(work, ec);
            return error;
        };
        for (auto& [name, file] : entries) {
            const std::string label = src.string() + "[" + name + "]";
            if (isLayerFile(name)) {
                LayerJob job;
                job.src = resolvedPath(file);
                job.dst = work / "out" / name;
                fs::create_directories(job.dst.parent_path(), ec);
                job.srcDir = job.src.parent_path();
                job.dstDir = resolvedPath(job.dst.parent_path());
                job.label = label;
                job.anchor = false;
                job.follow = false;
                if (auto r = layer(job); !r) return failed(std::move(r).error());
                file = job.dst;
            } else if (isCloudFile(name)) {
                const std::string renamed = asAthc(name);
                const fs::path to = work / "out" / renamed;
                fs::create_directories(to.parent_path(), ec);
                auto converted = lod::migrateLrtc(file, to);
                if (!converted) return failed(std::move(converted).error());
                if (*converted) {
                    note(MigrateChange::Kind::Asset, label, "", name, renamed);
                }
                name = renamed;
                file = to;
            }
        }
        {
            SdfZipFileWriter writer = SdfZipFileWriter::CreateNew(dst.string());
            if (!writer) {
                return failed(Error(ErrorCode::IoFailure, "cannot write " + dst.string()));
            }
            for (const auto& [name, file] : entries) {
                if (writer.AddFile(file.string(), name).empty()) {
                    writer.Discard();
                    return failed(Error(ErrorCode::IoFailure, "cannot add " + name + " to " + dst.string()));
                }
            }
            if (!writer.Save()) {
                return failed(Error(ErrorCode::IoFailure, "cannot write " + dst.string()));
            }
        }
        fs::remove_all(work, ec);
        return ok();
    }
};

}   // namespace

size_t MigrateReport::count(MigrateChange::Kind kind) const {
    return static_cast<size_t>(
        std::count_if(changes.begin(), changes.end(), [kind](const MigrateChange& c) { return c.kind == kind; }));
}

size_t MigrateReport::renames() const {
    return static_cast<size_t>(std::count_if(changes.begin(), changes.end(), [](const MigrateChange& c) {
        return c.kind != MigrateChange::Kind::File && c.kind != MigrateChange::Kind::Warning;
    }));
}

const char* toString(MigrateChange::Kind kind) noexcept {
    switch (kind) {
    case MigrateChange::Kind::Schema: return "schema";
    case MigrateChange::Kind::Property: return "property";
    case MigrateChange::Kind::Target: return "target";
    case MigrateChange::Kind::Value: return "value";
    case MigrateChange::Kind::Metadata: return "metadata";
    case MigrateChange::Kind::Asset: return "asset";
    case MigrateChange::Kind::Anchored: return "anchored";
    case MigrateChange::Kind::File: return "file";
    case MigrateChange::Kind::Warning: return "warning";
    }
    return "?";
}

Result<MigrateReport> migrate(const std::filesystem::path& in, const std::filesystem::path& out,
                              const MigrateOptions& options) {
    return Migrator(options).run(in, out);
}

}   // namespace athenea::usd
