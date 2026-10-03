// Copyright (c) 2026 jesus luque.
#include "BindingPurposes.h"

#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/sceneIndexPrimView.h>

#include <atomic>

#include "athenea/core/Log.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

/// A prim whose bindings are the one its purposes pick, under the
/// all-purpose name every purpose falls back to.
class ResolvedPrim final : public HdContainerDataSource {
public:
    HD_DECLARE_DATASOURCE(ResolvedPrim);

    ResolvedPrim(const HdContainerDataSourceHandle& input, const TfTokenVector& purposes, const SdfPath& path)
        : _input(input), _purposes(purposes), _path(path) {}

    TfTokenVector GetNames() override { return _input->GetNames(); }

    HdDataSourceBaseHandle Get(const TfToken& name) override {
        HdDataSourceBaseHandle data = _input->Get(name);
        if (name != HdMaterialBindingsSchema::GetSchemaToken()) {
            return data;
        }
        const HdContainerDataSourceHandle bindings = HdContainerDataSource::Cast(data);
        if (!bindings) {
            static std::atomic<int> told{0};
            if (told.fetch_add(1) < 6) {
                const HdContainerDataSourceHandle raw =
                    HdContainerDataSource::Cast(_input->Get(TfToken("usdMaterialBindings")));
                std::string names;
                if (raw) {
                    for (const TfToken& name : raw->GetNames()) {
                        names += (names.empty() ? "" : ", ") + name.GetString();
                    }
                }
                athenea::log::debug("hdAthenea: {} has no resolved bindings; usdMaterialBindings {}", _path.GetString(),
                                raw ? ("carries [" + names + "]") : std::string("is absent too"));
            }
            return data;
        }
        for (const TfToken& purpose : _purposes) {
            if (HdDataSourceBaseHandle binding = bindings->Get(purpose)) {
                return HdRetainedContainerDataSource::New(HdMaterialBindingsSchemaTokens->allPurpose, binding);
            }
        }
        // Nothing this renderer asked for: what the prim does carry, once, so
        // a stage that arrives unbound says which purposes it was bound with.
        std::string names;
        for (const TfToken& name : bindings->GetNames()) {
            names += (names.empty() ? "" : ", ") + name.GetString();
        }
        athenea::log::debug("hdAthenea: material bindings of no purpose we asked for; the prim carries [{}]", names);
        return HdRetainedContainerDataSource::New();
    }

private:
    HdContainerDataSourceHandle _input;
    TfTokenVector               _purposes;
    SdfPath                     _path;
};

}   // namespace

HdAtheneaBindingPurposesSceneIndexRefPtr HdAtheneaBindingPurposesSceneIndex::New(const HdSceneIndexBaseRefPtr& input,
                                                                         const TfTokenVector& purposes) {
    return TfCreateRefPtr(new HdAtheneaBindingPurposesSceneIndex(input, purposes));
}

HdAtheneaBindingPurposesSceneIndex::HdAtheneaBindingPurposesSceneIndex(const HdSceneIndexBaseRefPtr& input,
                                                               const TfTokenVector& purposes)
    : HdSingleInputFilteringSceneIndexBase(input), _purposes(purposes) {}

void HdAtheneaBindingPurposesSceneIndex::SetPurposes(const TfTokenVector& purposes) {
    if (purposes == _purposes) {
        return;
    }
    _purposes = purposes;
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return;
    }
    // Walking the scene's paths is bookkeeping: which prims carry bindings.
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    for (const SdfPath& path : HdSceneIndexPrimView(input)) {
        const HdSceneIndexPrim prim = input->GetPrim(path);
        if (prim.dataSource && prim.dataSource->Get(HdMaterialBindingsSchema::GetSchemaToken())) {
            dirtied.emplace_back(path, HdMaterialBindingsSchema::GetDefaultLocator());
        }
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
}

void HdAtheneaBindingPurposesSceneIndex::DirtyAll(const HdDataSourceLocatorSet& locators) {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return;
    }
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    for (const SdfPath& path : HdSceneIndexPrimView(input)) {
        dirtied.emplace_back(path, locators);
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
}

HdSceneIndexPrim HdAtheneaBindingPurposesSceneIndex::GetPrim(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    if (!input) {
        return {TfToken(), nullptr};
    }
    HdSceneIndexPrim prim = input->GetPrim(primPath);
    if (prim.dataSource) {
        prim.dataSource = ResolvedPrim::New(prim.dataSource, _purposes, primPath);
    }
    return prim;
}

SdfPathVector HdAtheneaBindingPurposesSceneIndex::GetChildPrimPaths(const SdfPath& primPath) const {
    const HdSceneIndexBaseRefPtr input = _GetInputSceneIndex();
    return input ? input->GetChildPrimPaths(primPath) : SdfPathVector();
}

void HdAtheneaBindingPurposesSceneIndex::_PrimsAdded(const HdSceneIndexBase&,
                                                 const HdSceneIndexObserver::AddedPrimEntries& entries) {
    _SendPrimsAdded(entries);
}

void HdAtheneaBindingPurposesSceneIndex::_PrimsRemoved(const HdSceneIndexBase&,
                                                   const HdSceneIndexObserver::RemovedPrimEntries& entries) {
    _SendPrimsRemoved(entries);
}

void HdAtheneaBindingPurposesSceneIndex::_PrimsDirtied(const HdSceneIndexBase&,
                                                   const HdSceneIndexObserver::DirtiedPrimEntries& entries) {
    _SendPrimsDirtied(entries);
}

PXR_NAMESPACE_CLOSE_SCOPE
