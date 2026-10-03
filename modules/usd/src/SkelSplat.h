// Copyright (c) 2026 jesus luque.
//
// A splat cloud carried by a UsdSkel Skeleton, the way the specification
// binds one: `SkelBindingAPI` on the ParticleField, `skel:skeleton` naming
// the Skeleton, `skel:animationSource` on it (or on the binding). UsdSkel's
// own imaging resolves the Skeleton -- its `resolvedSkeleton` carries the
// skinning transforms at any time -- but skins only PointBased prims, and a
// ParticleField is not one. This filter is the missing half for clouds: it
// puts the resolved transforms on the cloud as the primvar the engine reads
// (`athenea:splat:skinningXforms`, remapped to the cloud's own `skel:joints`
// order), and tells the cloud when its Skeleton moved. A cloud that authors
// that primvar itself -- the cache `athenea mesh2splat --skinned` writes -- keeps
// it: the cache is why a film of sixty joints costs kilobytes a frame.
#pragma once

#include <pxr/imaging/hd/filteringSceneIndex.h>

#include <map>
#include <set>

PXR_NAMESPACE_OPEN_SCOPE

TF_DECLARE_REF_PTRS(HdAtheneaSkelSplatSceneIndex);

class HdAtheneaSkelSplatSceneIndex final : public HdSingleInputFilteringSceneIndexBase {
public:
    static HdAtheneaSkelSplatSceneIndexRefPtr New(const HdSceneIndexBaseRefPtr& input);

    HdSceneIndexPrim GetPrim(const SdfPath& primPath) const override;
    SdfPathVector GetChildPrimPaths(const SdfPath& primPath) const override;

protected:
    explicit HdAtheneaSkelSplatSceneIndex(const HdSceneIndexBaseRefPtr& input);

    void _PrimsAdded(const HdSceneIndexBase& sender, const HdSceneIndexObserver::AddedPrimEntries& entries) override;
    void _PrimsRemoved(const HdSceneIndexBase& sender, const HdSceneIndexObserver::RemovedPrimEntries& entries) override;
    void _PrimsDirtied(const HdSceneIndexBase& sender, const HdSceneIndexObserver::DirtiedPrimEntries& entries) override;

private:
    /// The Skeleton a cloud at `path` is bound to, out of the input, or an
    /// empty path. Bookkeeping: which prim to watch for which.
    SdfPath _skeletonOf(const SdfPath& path) const;
    void _bind(const SdfPath& cloud, const SdfPath& skeleton);
    void _unbind(const SdfPath& cloud);

    std::map<SdfPath, std::set<SdfPath>> _cloudsOf;    ///< skeleton -> the clouds it carries
    std::map<SdfPath, SdfPath>           _carriedBy;   ///< cloud -> its skeleton
};

PXR_NAMESPACE_CLOSE_SCOPE
