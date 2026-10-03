// Copyright (c) 2026 jesus luque.
#include "SkelSplat.h"

#include <pxr/imaging/hd/overlayContainerDataSource.h>
#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/usd/usdSkel/animMapper.h>
#include <pxr/usdImaging/usdSkelImaging/bindingSchema.h>
#include <pxr/usdImaging/usdSkelImaging/resolvedSkeletonSchema.h>
#include <pxr/usdImaging/usdSkelImaging/skeletonSchema.h>

#include "athenea/core/Log.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

const TfToken& cacheToken() {
    static const TfToken token("athenea:splat:skinningXforms");
    return token;
}

/// The Skeleton's skinning transforms in the cloud's joint order: what
/// `skel:joints` on the binding asks for, where it names a subset or another
/// order of the Skeleton's joints. Sampled as the Skeleton's are, so a
/// shutter reads the same instants from either.
class RemappedXforms final : public HdMatrix4fArrayDataSource {
public:
    HD_DECLARE_DATASOURCE(RemappedXforms);

    VtValue GetValue(Time shutterOffset) override { return VtValue(GetTypedValue(shutterOffset)); }
    VtMatrix4fArray GetTypedValue(Time shutterOffset) override {
        VtMatrix4fArray out;
        _mapper.RemapTransforms(_source->GetTypedValue(shutterOffset), &out);
        return out;
    }
    bool GetContributingSampleTimesForInterval(Time start, Time end, std::vector<Time>* times) override {
        return _source->GetContributingSampleTimesForInterval(start, end, times);
    }

private:
    RemappedXforms(UsdSkelAnimMapper mapper, HdMatrix4fArrayDataSourceHandle source)
        : _mapper(std::move(mapper)), _source(std::move(source)) {}

    UsdSkelAnimMapper                _mapper;
    HdMatrix4fArrayDataSourceHandle  _source;
};

SdfPath boundSkeleton(const HdContainerDataSourceHandle& prim) {
    const UsdSkelImagingBindingSchema binding = UsdSkelImagingBindingSchema::GetFromParent(prim);
    const HdPathDataSourceHandle skeleton = binding.GetSkeleton();
    return skeleton ? skeleton->GetTypedValue(0.0F) : SdfPath();
}

}   // namespace

HdAtheneaSkelSplatSceneIndexRefPtr HdAtheneaSkelSplatSceneIndex::New(const HdSceneIndexBaseRefPtr& input) {
    return TfCreateRefPtr(new HdAtheneaSkelSplatSceneIndex(input));
}

HdAtheneaSkelSplatSceneIndex::HdAtheneaSkelSplatSceneIndex(const HdSceneIndexBaseRefPtr& input)
    : HdSingleInputFilteringSceneIndexBase(input) {}

HdSceneIndexPrim HdAtheneaSkelSplatSceneIndex::GetPrim(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return {TfToken(), nullptr};
    }
    HdSceneIndexPrim prim = input->GetPrim(primPath);
    if (prim.primType != HdPrimTypeTokens->particleField || !prim.dataSource) {
        return prim;
    }
    const SdfPath skeletonPath = boundSkeleton(prim.dataSource);
    if (skeletonPath.IsEmpty()) {
        return prim;
    }
    // The cloud's own cache of the transforms wins: it is the same numbers,
    // already written down, and nothing to resolve at each instant.
    const HdPrimvarsSchema primvars = HdPrimvarsSchema::GetFromParent(prim.dataSource);
    if (primvars.GetPrimvar(cacheToken()).GetPrimvarValue()) {
        return prim;
    }
    const HdContainerDataSourceHandle skeleton = input->GetPrim(skeletonPath).dataSource;
    if (!skeleton) {
        return prim;
    }
    const UsdSkelImagingResolvedSkeletonSchema resolved = UsdSkelImagingResolvedSkeletonSchema::GetFromParent(skeleton);
    HdMatrix4fArrayDataSourceHandle xforms = resolved.GetSkinningTransforms();
    if (!xforms) {
        return prim;
    }
    // The order the cloud's indices count joints in: the binding's
    // `skel:joints` where it has them, the Skeleton's otherwise.
    const UsdSkelImagingBindingSchema binding = UsdSkelImagingBindingSchema::GetFromParent(prim.dataSource);
    if (const HdTokenArrayDataSourceHandle joints = binding.GetJoints()) {
        const VtTokenArray own = joints->GetTypedValue(0.0F);
        const HdTokenArrayDataSourceHandle skeletonJoints =
            UsdSkelImagingSkeletonSchema::GetFromParent(skeleton).GetJoints();
        if (!own.empty() && skeletonJoints) {
            UsdSkelAnimMapper mapper(skeletonJoints->GetTypedValue(0.0F), own);
            if (!mapper.IsNull() && !mapper.IsIdentity()) {
                xforms = RemappedXforms::New(std::move(mapper), xforms);
            }
        }
    }
    const HdContainerDataSourceHandle primvar =
        HdPrimvarSchema::Builder()
            .SetPrimvarValue(xforms)
            .SetInterpolation(HdPrimvarSchema::BuildInterpolationDataSource(HdPrimvarSchemaTokens->constant))
            .Build();
    prim.dataSource = HdOverlayContainerDataSource::New(
        HdRetainedContainerDataSource::New(HdPrimvarsSchema::GetSchemaToken(),
                                           HdRetainedContainerDataSource::New(cacheToken(), primvar)),
        prim.dataSource);
    return prim;
}

SdfPathVector HdAtheneaSkelSplatSceneIndex::GetChildPrimPaths(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetChildPrimPaths(primPath) : SdfPathVector();
}

SdfPath HdAtheneaSkelSplatSceneIndex::_skeletonOf(const SdfPath& path) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return SdfPath();
    }
    const HdSceneIndexPrim prim = input->GetPrim(path);
    if (prim.primType != HdPrimTypeTokens->particleField || !prim.dataSource) {
        return SdfPath();
    }
    return boundSkeleton(prim.dataSource);
}

void HdAtheneaSkelSplatSceneIndex::_bind(const SdfPath& cloud, const SdfPath& skeleton) {
    _unbind(cloud);
    if (skeleton.IsEmpty()) {
        return;
    }
    _carriedBy[cloud] = skeleton;
    _cloudsOf[skeleton].insert(cloud);
}

void HdAtheneaSkelSplatSceneIndex::_unbind(const SdfPath& cloud) {
    const auto held = _carriedBy.find(cloud);
    if (held == _carriedBy.end()) {
        return;
    }
    if (const auto clouds = _cloudsOf.find(held->second); clouds != _cloudsOf.end()) {
        clouds->second.erase(cloud);
        if (clouds->second.empty()) {
            _cloudsOf.erase(clouds);
        }
    }
    _carriedBy.erase(held);
}

void HdAtheneaSkelSplatSceneIndex::_PrimsAdded(const HdSceneIndexBase&,
                                           const HdSceneIndexObserver::AddedPrimEntries& entries) {
    HdSceneIndexObserver::DirtiedPrimEntries moved;
    for (const HdSceneIndexObserver::AddedPrimEntry& entry : entries) {
        if (entry.primType == HdPrimTypeTokens->particleField) {
            _bind(entry.primPath, _skeletonOf(entry.primPath));
        }
        // A Skeleton that arrives after the clouds it carries: they read it
        // now, whatever they read before.
        if (const auto clouds = _cloudsOf.find(entry.primPath); clouds != _cloudsOf.end()) {
            for (const SdfPath& cloud : clouds->second) {
                moved.emplace_back(cloud, HdPrimvarsSchema::GetDefaultLocator().Append(cacheToken()));
            }
        }
    }
    _SendPrimsAdded(entries);
    if (!moved.empty()) {
        _SendPrimsDirtied(moved);
    }
}

void HdAtheneaSkelSplatSceneIndex::_PrimsRemoved(const HdSceneIndexBase&,
                                             const HdSceneIndexObserver::RemovedPrimEntries& entries) {
    for (const HdSceneIndexObserver::RemovedPrimEntry& entry : entries) {
        for (auto it = _carriedBy.begin(); it != _carriedBy.end();) {
            const SdfPath cloud = it->first;
            ++it;
            if (cloud.HasPrefix(entry.primPath)) {
                _unbind(cloud);
            }
        }
    }
    _SendPrimsRemoved(entries);
}

void HdAtheneaSkelSplatSceneIndex::_PrimsDirtied(const HdSceneIndexBase&,
                                             const HdSceneIndexObserver::DirtiedPrimEntries& entries) {
    HdSceneIndexObserver::DirtiedPrimEntries moved;
    for (const HdSceneIndexObserver::DirtiedPrimEntry& entry : entries) {
        // A cloud whose binding changed: bound anew, and told to read again.
        if (entry.dirtyLocators.Intersects(UsdSkelImagingBindingSchema::GetDefaultLocator()) &&
            _skeletonOf(entry.primPath) != SdfPath()) {
            _bind(entry.primPath, _skeletonOf(entry.primPath));
            moved.emplace_back(entry.primPath, HdPrimvarsSchema::GetDefaultLocator().Append(cacheToken()));
        }
        // A Skeleton that moved: every cloud it carries has new joints. This
        // is the notice a time change comes as, since the cloud's own
        // primvars are not sampled and nothing else about it changed.
        if (entry.dirtyLocators.Intersects(UsdSkelImagingResolvedSkeletonSchema::GetDefaultLocator())) {
            if (const auto clouds = _cloudsOf.find(entry.primPath); clouds != _cloudsOf.end()) {
                for (const SdfPath& cloud : clouds->second) {
                    moved.emplace_back(cloud, HdPrimvarsSchema::GetDefaultLocator().Append(cacheToken()));
                }
            }
        }
    }
    _SendPrimsDirtied(entries);
    if (!moved.empty()) {
        _SendPrimsDirtied(moved);
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
