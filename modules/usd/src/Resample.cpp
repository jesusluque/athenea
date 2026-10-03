// Copyright (c) 2026 jesus luque.
#include "Resample.h"

#include <set>
#include <vector>

#include <pxr/imaging/hd/sceneIndexPrimView.h>

PXR_NAMESPACE_OPEN_SCOPE

HdAtheneaResampleSceneIndexRefPtr HdAtheneaResampleSceneIndex::New(const HdSceneIndexBaseRefPtr& input) {
    return TfCreateRefPtr(new HdAtheneaResampleSceneIndex(input));
}

HdAtheneaResampleSceneIndex::HdAtheneaResampleSceneIndex(const HdSceneIndexBaseRefPtr& input)
    : HdSingleInputFilteringSceneIndexBase(input) {}

void HdAtheneaResampleSceneIndex::DirtyAll(const HdDataSourceLocatorSet& locators, const SdfPath& root) {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return;
    }
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    for (const SdfPath& path : HdSceneIndexPrimView(input, root)) {
        dirtied.emplace_back(path, locators);
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
}

HdSceneIndexPrim HdAtheneaResampleSceneIndex::GetPrim(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetPrim(primPath) : HdSceneIndexPrim{TfToken(), nullptr};
}

SdfPathVector HdAtheneaResampleSceneIndex::GetChildPrimPaths(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetChildPrimPaths(primPath) : SdfPathVector();
}

void HdAtheneaResampleSceneIndex::_PrimsAdded(const HdSceneIndexBase&, const HdSceneIndexObserver::AddedPrimEntries& entries) {
    _SendPrimsAdded(entries);
}

void HdAtheneaResampleSceneIndex::_PrimsRemoved(const HdSceneIndexBase&,
                                            const HdSceneIndexObserver::RemovedPrimEntries& entries) {
    _SendPrimsRemoved(entries);
}

void HdAtheneaResampleSceneIndex::_PrimsDirtied(const HdSceneIndexBase&,
                                            const HdSceneIndexObserver::DirtiedPrimEntries& entries) {
    _SendPrimsDirtied(entries);
}

size_t HdAtheneaResampleUpstream(const HdSceneIndexBaseRefPtr& terminal, const HdDataSourceLocatorSet& locators,
                             const SdfPath& root) {
    std::vector<HdSceneIndexBaseRefPtr> pending{terminal};
    std::set<const HdSceneIndexBase*> seen;
    std::vector<HdAtheneaResampleSceneIndexRefPtr> found;
    while (!pending.empty()) {
        const HdSceneIndexBaseRefPtr scene = pending.back();
        pending.pop_back();
        if (!scene || !seen.insert(get_pointer(scene)).second) {
            continue;
        }
        if (auto resample = TfDynamic_cast<HdAtheneaResampleSceneIndexRefPtr>(scene)) {
            found.push_back(resample);
        }
        if (auto filtering = TfDynamic_cast<HdFilteringSceneIndexBaseRefPtr>(scene)) {
            for (const HdSceneIndexBaseRefPtr& input : filtering->GetInputScenes()) {
                pending.push_back(input);
            }
        }
        if (HdEncapsulatingSceneIndexBase* encapsulating = HdEncapsulatingSceneIndexBase::Cast(scene)) {
            for (const HdSceneIndexBaseRefPtr& inner : encapsulating->GetEncapsulatedScenes()) {
                pending.push_back(inner);
            }
        }
    }
    for (const HdAtheneaResampleSceneIndexRefPtr& resample : found) {
        resample->DirtyAll(locators, root);
    }
    return found.size();
}

PXR_NAMESPACE_CLOSE_SCOPE
