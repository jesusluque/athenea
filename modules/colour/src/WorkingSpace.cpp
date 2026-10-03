// Copyright (c) 2026 jesus luque.
#include "athenea/colour/WorkingSpace.h"

namespace athenea::colour {

Result<gpu::WorkingSpace> workingSpaceRecord(const ColourNames& names, const std::string& name) {
    // Linear, as a working space is: an empty name is the default.
    const ResolvedSpace resolved = names.resolve(name, FileEncoding::Linear);
    if (resolved.kind == SpaceKind::Working) {
        return gpu::linRec709Scene();
    }
    const char* what = resolved.kind == SpaceKind::Raw       ? "data, not a colour space"
                       : resolved.kind == SpaceKind::Srgb    ? "sRGB-encoded, not linear"
                       : resolved.kind == SpaceKind::Unknown ? "a name no config here knows"
                                                             : "not linear Rec.709";
    return Error::make(ErrorCode::Unsupported, "colour: '{}' as the working space is {}; rendering in {}", name,
                       what, kWorkingSpace);
}

}   // namespace athenea::colour
