// Copyright (c) 2026 jesus luque.
//
// The config a ColourCompiler and its ColourNames share. Private: OCIO's
// headers stay inside the colour module.
#pragma once

#include <string>

#if ATHENEA_HAVE_OCIO
#include <OpenColorIO/OpenColorIO.h>
namespace OCIO = OCIO_NAMESPACE;
#endif

namespace athenea::colour {

struct ColourConfig {
    std::string uri;
    std::string cacheId;   ///< OCIO's for the config, or "table" without OCIO
    std::string label;     ///< the config's name
#if ATHENEA_HAVE_OCIO
    OCIO::ConstConfigRcPtr config;
    /// The built-in studio config, for a name the config in use does not
    /// know; the same pointer when that is the config in use.
    OCIO::ConstConfigRcPtr studio;
#endif
};

}   // namespace athenea::colour
