// Copyright (c) 2026 jesus luque.
//
// .athz on the command line: `athenea convert` into and out of it, and
// `athenea athz-census`, what each of a cloud's byte planes costs raw, under
// rANS and under gzip, and how long the device takes to decode its pages
// (docs/decisions.md, ".athz", E1).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <zlib.h>

#include "Commands.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/SplatWriters.h"
#include "athenea/lod/Athc.h"
#include "athenea/lod/Athz.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

namespace athenea::cli {
namespace fs = std::filesystem;
namespace {

bool hasExtension(const fs::path& path, std::initializer_list<const char*> extensions) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return std::any_of(extensions.begin(), extensions.end(), [&](const char* e) { return ext == e; });
}

/// Any cloud as one with levels, every chunk on the device in order.
Result<lod::LodCloud> loadTree(gpu::ShaderLibrary& library, scene::CloudLoader& loader, const fs::path& input,
                               uint32_t degree, uint32_t chunkSplats) {
    if (lod::isAthc(input)) {
        return lod::readAthc(library.device(), input);
    }
    if (lod::isAthz(input)) {
        return lod::readAthz(library, input);
    }
    Result<scene::GpuSplats> splats = hasExtension(input, {".usd", ".usda", ".usdc", ".usdz"})
                                          ? usd::loadParticleField(loader, input, {}, degree)
                                          : scene::loadSplatFile(loader, input, degree);
    if (!splats) return std::move(splats).error();
    auto builder = lod::LodBuilder::create(library);
    if (!builder) return std::move(builder).error();
    lod::LodBuildSettings settings;
    settings.chunkSplats = chunkSplats;
    return builder->build(*splats, settings);
}

/// Bytes deflate (gzip's coding, without its 18 bytes of frame) makes of
/// `n` bytes at `level`: byte coding on the CPU, as the formats it stands
/// for do it.
uint64_t deflated(const uint8_t* data, uint64_t n, int level = 6) {
    z_stream z{};
    if (deflateInit(&z, level) != Z_OK) {
        return 0;
    }
    std::vector<uint8_t> out(1 << 20);
    uint64_t total = 0;
    uint64_t at = 0;
    int flush = Z_NO_FLUSH;
    do {
        const uint64_t take = std::min<uint64_t>(n - at, uint64_t{1} << 30);
        z.next_in = const_cast<Bytef*>(data + at);
        z.avail_in = static_cast<uInt>(take);
        at += take;
        flush = at >= n ? Z_FINISH : Z_NO_FLUSH;
        do {
            z.next_out = out.data();
            z.avail_out = static_cast<uInt>(out.size());
            deflate(&z, flush);
            total += out.size() - z.avail_out;
        } while (z.avail_out == 0);
    } while (flush != Z_FINISH);
    deflateEnd(&z);
    return total;
}

uint64_t deflatedFile(const fs::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) return 0;
    return deflated(reinterpret_cast<const uint8_t*>(file->bytes().data()), file->size());
}

/// Words two clouds' splats and merged levels differ by, counted by a kernel.
Result<uint64_t> recordDifference(gpu::ShaderLibrary& library, const lod::LodCloud& a, const lod::LodCloud& b) {
    gpu::Device& device = library.device();
    auto compare = gpu::ComputeKernel::create(library, "athenea/lod/athz_check", "athzCompare");
    if (!compare) return std::move(compare).error();
    std::vector<uint32_t> zero(2, 0);
    auto counter = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(zero), "census.counter");
    if (!counter) return std::move(counter).error();
    gpu::CommandBatch batch(device);
    const auto words = [&](const gpu::Buffer& x, const gpu::Buffer& y, uint64_t n) {
        if (n == 0 || !x.valid() || !y.valid()) return;
        compare->dispatch(batch, {static_cast<uint32_t>(n), 1, 1}, [&](rhi::ShaderCursor c) {
            c["a"].setBinding(x.rhi());
            c["b"].setBinding(y.rhi());
            c["counters"].setBinding(counter->rhi());
            c["positions"].setBinding(x.rhi());
            c["shape"].setBinding(x.rhi());
            c["sh"].setBinding(x.rhi());
            c["flat"].setBinding(counter->rhi());
            c["params"]["words"].setData(static_cast<uint32_t>(n));
            c["params"]["offsetA"].setData(0u);
            c["params"]["offsetB"].setData(0u);
        });
    };
    const auto cloud = [&](const scene::GpuSplats& x, const scene::GpuSplats& y, uint64_t n) {
        words(x.positions, y.positions, n * 4);
        words(x.shape, y.shape, n * 4);
        words(x.sh, y.sh, n * x.shWords);
        words(x.normals, y.normals, n);
        words(x.emission, y.emission, n);
        words(x.pbr, y.pbr, n);
        words(x.lobes, y.lobes, n * 3);
        words(x.transfer, y.transfer, n * x.transferWords);
        words(x.shadowBits, y.shadowBits, n * x.shadowWords);
    };
    cloud(a.splats, b.splats, a.count);
    words(a.groups, b.groups, a.count);
    for (size_t l = 0; l < std::min(a.levels.size(), b.levels.size()); ++l) {
        cloud(a.levels[l].gaussians, b.levels[l].gaussians, a.levels[l].gaussians.count);
        words(a.levels[l].cells, b.levels[l].cells, a.levels[l].gaussians.count);
    }
    ATHENEA_TRY(batch.submit(true));
    uint32_t n = 0;
    ATHENEA_TRY(counter->read(device, 0, 4, &n));
    return n;
}

/// The names of an element's bytes, field by field (Athz.cpp's order).
std::vector<std::string> byteNames(const lod::LodCloud& cloud, uint32_t planes) {
    std::vector<std::string> names;
    const auto add = [&](const std::string& field, uint32_t words) {
        for (uint32_t w = 0; w < words; ++w) {
            for (uint32_t b = 0; b < 4; ++b) {
                names.push_back(field + (words > 1 ? "[" + std::to_string(w) + "]" : "") + "." + std::to_string(b));
            }
        }
    };
    const scene::GpuSplats& s = cloud.splats;
    add("position", 4);
    add("shape", 4);
    add("sh", s.shWords);
    add("tail", 1);
    if (s.hasNormals()) add("normal", 1);
    if (s.hasEmission()) add("emission", 1);
    if (s.pbr.valid()) add("pbr", 1);
    if (s.lobes.valid()) add("lobes", 3);
    if (s.transfer.valid()) add("transfer", s.transferWords);
    if (s.shadowBits.valid()) add("shadow", s.shadowWords);
    names.resize(planes, "?");
    return names;
}

/// A byte plane's field: its name up to the word.
std::string fieldOf(const std::string& name) {
    const size_t cut = name.find_first_of("[.");
    return name.substr(0, cut);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double mb(uint64_t bytes) {
    return static_cast<double>(bytes) / 1.0e6;
}


struct CpcaCensus {
    uint64_t raw = 0, residual = 0;
    double   agree = 0.0;
};

/// The splats' transfer as CPCA weights, at order 0 as they are and as a
/// residual from the finest merged group's (athz_cpca.slang).
Result<CpcaCensus> cpcaCensus(gpu::ShaderLibrary& library, const lod::LodCloud& tree, const fs::path& basisPath) {
    gpu::Device& device = library.device();
    constexpr uint32_t kC = 256, kK = 8;
    const uint32_t dims[3] = {16, 48, 48};
    uint64_t floats = uint64_t{kC} * 112 + 3;
    uint32_t meanAt[3] = {0, 0, 0};
    for (int b = 0; b < 3; ++b) {
        meanAt[b] = static_cast<uint32_t>(floats);
        floats += uint64_t{kC} * dims[b] + uint64_t{kC} * dims[b] * kK + uint64_t{kC} * kK;
    }
    auto file = platform::MappedFile::open(basisPath);
    if (!file) return std::move(file).error();
    if (file->size() != floats * 4) {
        return Error::make(ErrorCode::InvalidArgument, "{} is not a basis of 256 clusters and 8 axes", basisPath.string());
    }
    const auto make = [&](uint64_t bytes, const void* data, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(bytes, 4);
        desc.elementBytes = 4;
        desc.label = label;
        return gpu::Buffer::create(device, desc, data);
    };
    const lod::LodLevel& finest = tree.levels.back();
    const uint32_t splats = tree.count, groups = finest.gaussians.count;
    const uint32_t codeWords = 1 + 3 * kK;
    auto basis = make(file->size(), file->bytes().data(), "census.basis");
    auto codes = make(uint64_t{splats + groups} * codeWords * 4, nullptr, "census.codes");
    const uint32_t histGroups = 3 + 3 * 3 * kK;
    std::vector<uint32_t> zero(uint64_t{histGroups} * 256, 0);
    auto hist = make(zero.size() * 4, zero.data(), "census.hist");
    auto bytes = make(uint64_t{histGroups} * 4, zero.data(), "census.bytes");
    if (!basis || !codes || !hist || !bytes) return Error(ErrorCode::OutOfMemory, "census: the CPCA buffers");
    auto cpca = gpu::ComputeKernel::create(library, "athenea/lod/athz_cpca", "athzCpca");
    auto histogram = gpu::ComputeKernel::create(library, "athenea/lod/athz_cpca", "athzCpcaHist");
    auto bits = gpu::ComputeKernel::create(library, "athenea/lod/athz_cpca", "athzCpcaBits");
    if (!cpca) return std::move(cpca).error();
    if (!histogram) return std::move(histogram).error();
    if (!bits) return std::move(bits).error();
    const auto bind = [&](rhi::ShaderCursor c, uint32_t count) {
        c["params"]["count"].setData(count);
        c["params"]["clusters"].setData(kC);
        c["params"]["components"].setData(kK);
        c["params"]["levels"].setData(7.0F);
        c["params"]["meanAt0"].setData(meanAt[0]);
        c["params"]["meanAt1"].setData(meanAt[1]);
        c["params"]["meanAt2"].setData(meanAt[2]);
        c["params"]["normAt"].setData(kC * 112);
        c["params"]["transferWords"].setData(tree.splats.transferWords);
        c["params"]["codeWords"].setData(codeWords);
        c["params"]["parentBase"].setData(splats);
        c["params"]["groups"].setData(histGroups);
        c["basis"].setBinding(basis->rhi());
        c["codes"].setBinding(codes->rhi());
        c["histogram"].setBinding(hist->rhi());
        c["bytes"].setBinding(bytes->rhi());
        c["groupOf"].setBinding(tree.groups.rhi());
    };
    gpu::CommandBatch batch(device);
    // The splats' codes first, then the finest groups' after them.
    cpca->dispatch(batch, {splats, 1, 1}, [&](rhi::ShaderCursor c) {
        bind(c, splats);
        c["transfer"].setBinding(tree.splats.transfer.rhi());
    });
    ATHENEA_TRY(batch.submit(true));
    auto parents = make(uint64_t{groups} * codeWords * 4, nullptr, "census.parentCodes");
    if (!parents) return Error(ErrorCode::OutOfMemory, "census: the CPCA buffers");
    cpca->dispatch(batch, {groups, 1, 1}, [&](rhi::ShaderCursor c) {
        bind(c, groups);
        c["transfer"].setBinding(finest.gaussians.transfer.rhi());
        c["codes"].setBinding(parents->rhi());
    });
    batch.encoder()->copyBuffer(codes->rhi(), uint64_t{splats} * codeWords * 4, parents->rhi(), 0,
                                uint64_t{groups} * codeWords * 4);
    batch.markDirty();
    histogram->dispatch(batch, {splats, 1, 1}, [&](rhi::ShaderCursor c) {
        bind(c, splats);
        c["transfer"].setBinding(tree.splats.transfer.rhi());
    });
    bits->dispatch(batch, {histGroups, 1, 1}, [&](rhi::ShaderCursor c) {
        bind(c, splats);
        c["transfer"].setBinding(tree.splats.transfer.rhi());
    });
    ATHENEA_TRY(batch.submit(true));
    auto b = bytes->readAll<uint32_t>(device);
    if (!b) return std::move(b).error();
    std::vector<uint32_t> flag(2);
    ATHENEA_TRY(hist->read(device, 512 * 4, 8, flag.data()));
    CpcaCensus out;
    const uint32_t w = 3 * kK;
    out.raw = (*b)[0];
    out.residual = (*b)[1] + (*b)[2];
    for (uint32_t a = 0; a < w; ++a) {
        out.raw += (*b)[3 + a];
        out.residual += (*b)[3 + w + a] + (*b)[3 + 2 * w + a];
    }
    out.agree = static_cast<double>(flag[0]) / std::max<double>(1.0, flag[0] + flag[1]);
    return out;
}

}   // namespace

bool convertWithAthz(const std::string& input, const std::string& output, const AthzConvertOptions& o) {
    const bool toAthz = lod::isAthz(output);
    const bool fromAthz = lod::isAthz(input);
    const bool toSpz = hasExtension(output, {".spz"});
    const bool toAthc = lod::isAthc(output);
    if (!toAthz && !(fromAthz && (toSpz || toAthc)) && !(lod::isAthc(input) && toSpz)) {
        return false;
    }
    auto device = gpu::Device::create();
    if (!device) {
        fail(device.error());
    }
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) {
        fail(loader.error());
    }
    if (toAthz) {
        auto tree = loadTree(library, *loader, input, o.degree, o.chunkSplats);
        if (!tree) fail(tree.error());
        lod::AthzSettings s;
        s.runGaussians = o.run;
        s.pageRuns = o.pageRuns;
        s.tableLog = o.tableLog;
        s.rans = !o.raw;
        s.source = fs::path(input).filename().string();
        auto stats = lod::writeAthz(library, *tree, output, s);
        if (!stats) fail(stats.error());
        out("wrote %s: %u splats and %llu merged in %u pages, %.2f MB (%.2f B a splat; G %.2f A %.2f T %.2f MB)\n",
            output.c_str(), tree->count, static_cast<unsigned long long>(stats->elements[0]), stats->pages,
            mb(stats->fileBytes), static_cast<double>(stats->fileBytes) / tree->count, mb(stats->layerBytes[0]),
            mb(stats->layerBytes[1]), mb(stats->layerBytes[2]));
        return true;
    }
    if (toAthc) {
        auto tree = lod::readAthz(library, input);
        if (!tree) fail(tree.error());
        if (auto written = lod::writeAthc(**device, *tree, output); !written) fail(written.error());
        out("wrote %s\n", output.c_str());
        return true;
    }
    // To SPZ: the splats of the file, decoded on the device -- of a .athz cut
    // to a prefix of its bands, the finest band it holds whole.
    Result<scene::GpuSplats> splats = Error(ErrorCode::Unknown, "");
    uint32_t band = 0;
    if (fromAthz) {
        splats = lod::readAthzCut(library, input, &band);
    } else {
        auto tree = lod::readAthc(**device, input);
        if (!tree) fail(tree.error());
        tree->splats.count = tree->count;
        splats = std::move(tree->splats);
    }
    if (!splats) fail(splats.error());
    if (auto written = lod::writeSpzOfCloud(library, *splats, output, o.spzVersion); !written) fail(written.error());
    out("wrote %s: %u splats%s\n", output.c_str(), splats->count,
        fromAthz ? (" (band " + std::to_string(band) + ")").c_str() : "");
    return true;
}

void addAthzCensus(CLI::App& app) {
    struct Options {
        std::vector<std::string> inputs;
        std::vector<uint32_t>    runs{64, 256};
        std::vector<uint32_t>    tableLogs{11, 12};
        std::string              dir;
        uint32_t                 degree = 3;
        uint32_t                 batchMiB = 8;
        uint32_t                 singlePages = 32;
        bool                     keep = false;
        std::string              transferBasis;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("athz-census",
                                   "what each byte plane of a cloud costs raw, under rANS and under gzip, and how long "
                                   "the device takes to decode its pages");
    cmd->add_option("inputs", o->inputs, ".athc / .athz / .usd / .ply / .spz / .sog")->required();
    cmd->add_option("--runs", o->runs, "elements a run, each tried");
    cmd->add_option("--table-logs", o->tableLogs, "rANS table sizes, log2, each tried");
    cmd->add_option("--dir", o->dir, "where the files measured are written");
    cmd->add_option("--degree", o->degree, "harmonic degree cap 0..3");
    cmd->add_option("--batch-mib", o->batchMiB, "coded MiB a decode dispatch");
    cmd->add_option("--single-pages", o->singlePages, "pages decoded one a dispatch");
    cmd->add_flag("--keep", o->keep, "keep the files measured");
    cmd->add_option("--transfer-basis", o->transferBasis,
                    "a clustered PCA basis (f32, 256 clusters, 8 axes): the transfer's weights measured too");
    cmd->callback([o] {
        auto device = gpu::Device::create();
        if (!device) fail(device.error());
        gpu::ShaderLibrary library(*device);
        auto loader = scene::CloudLoader::create(library);
        if (!loader) fail(loader.error());
        const fs::path dir = o->dir.empty() ? fs::temp_directory_path() : fs::path(o->dir);
        std::error_code ec;
        fs::create_directories(dir, ec);
        for (const std::string& input : o->inputs) {
            auto tree = loadTree(library, *loader, input, o->degree, uint32_t{1} << 16);
            if (!tree) fail(tree.error());
            const std::string stem = fs::path(input).stem().string();
            uint64_t merged = 0;
            for (const lod::LodLevel& l : tree->levels) merged += l.gaussians.count;
            out("\n## %s: %u splats, %llu merged in %zu levels, %u bytes an element\n\n", input.c_str(), tree->count,
                static_cast<unsigned long long>(merged), tree->levels.size(),
                static_cast<uint32_t>(tree->splats.shWords * 4 + 36 + (tree->splats.hasNormals() ? 4 : 0) +
                                      (tree->splats.hasEmission() ? 4 : 0) + (tree->splats.pbr.valid() ? 4 : 0) +
                                      (tree->splats.lobes.valid() ? 12 : 0) + tree->splats.transferWords * 4 +
                                      tree->splats.shadowWords * 4));
            const double elements = static_cast<double>(tree->count) + static_cast<double>(merged);
            // Today's: the .athc, and gzip of it.
            const fs::path athc = dir / (stem + ".census.athc");
            if (auto w = lod::writeAthc(**device, *tree, athc); !w) fail(w.error());
            const uint64_t athcBytes = fs::file_size(athc);
            const uint64_t gzipBytes = deflatedFile(athc);
            // Byte planes, gzip each (the paper's byte planes).
            lod::AthzSettings planesSettings;
            planesSettings.bytePlanes = true;
            const fs::path planesPath = dir / (stem + ".census.planes.athz");
            auto planesStats = lod::writeAthz(library, *tree, planesPath, planesSettings);
            if (!planesStats) fail(planesStats.error());
            auto planes = lod::athzBytePlanes(planesPath);
            if (!planes) fail(planes.error());
            std::array<std::vector<uint64_t>, 2> gzipPlane;
            uint64_t gzipPlanesTotal = 0;
            for (uint32_t k = 0; k < 2; ++k) {
                for (const std::vector<uint8_t>& p : (*planes)[k]) {
                    gzipPlane[k].push_back(p.empty() ? 0 : deflated(p.data(), p.size()));
                    gzipPlanesTotal += gzipPlane[k].back();
                }
            }
            const uint64_t tables = planesStats->tableBytes + planesStats->startsBytes;
            out("| coding | runs | table | file MB | B an element | vs gzip(.athc) |\n|---|---:|---:|---:|---:|---:|\n");
            const auto row = [&](const char* coding, uint32_t run, const std::string& table, uint64_t bytes) {
                out("| %s | %s | %s | %.2f | %.2f | %.3f |\n", coding, run ? std::to_string(run).c_str() : "-",
                    table.c_str(), mb(bytes), static_cast<double>(bytes) / elements,
                    static_cast<double>(bytes) / static_cast<double>(gzipBytes));
            };
            row(".athc", 0, "-", athcBytes);
            row("gzip(.athc)", 0, "-", gzipBytes);
            row("byte planes, gzip each", 0, "-", gzipPlanesTotal + tables);
            // Writes a file, checks it decodes bit for bit, keeps its stats.
            std::vector<std::tuple<std::string, uint32_t, uint32_t, fs::path>> timed;
            const auto measure = [&](const char* coding, uint32_t run, uint32_t n, bool rans, uint32_t transforms,
                                     bool contexts, bool time) -> lod::AthzStats {
                lod::AthzSettings s;
                s.runGaussians = run;
                s.pageRuns = 3072 / run;
                s.rans = rans;
                s.tableLog = n;
                s.transforms = transforms;
                s.measureContexts = contexts;
                const fs::path path = dir / (stem + ".census." + (rans ? "r" : "raw") + std::to_string(run) + "n" +
                                             std::to_string(n) + "t" + std::to_string(transforms) + ".athz");
                auto st = lod::writeAthz(library, *tree, path, s);
                if (!st) fail(st.error());
                row(coding, run, rans ? "2^" + std::to_string(n) : std::string("-"), st->fileBytes);
                auto back = lod::readAthz(library, path);
                if (!back) fail(back.error());
                auto differ = recordDifference(library, *tree, *back);
                if (!differ) fail(differ.error());
                if (*differ != 0) {
                    fail(Error::make(ErrorCode::InternalError, "{}: {} words decoded differently", path.string(), *differ));
                }
                if (time) {
                    timed.emplace_back(coding, run, n, path);
                } else if (!o->keep) {
                    fs::remove(path, ec);
                }
                return *st;
            };
            std::map<uint32_t, lod::AthzStats> raw;
            lod::AthzStats order0, transformed, transformed12;
            for (const uint32_t run : o->runs) {
                raw[run] = measure("athz raw bit-pack", run, 11, false, 1, false, false);
                for (const uint32_t n : o->tableLogs) {
                    auto st = measure("athz rANS order 0", run, n, true, 1, false, run == 256 && n == 11);
                    if (run == o->runs.back() && n == o->tableLogs.front()) order0 = st;
                }
            }
            for (const uint32_t run : o->runs) {
                for (const uint32_t n : o->tableLogs) {
                    const bool last = run == o->runs.back();
                    auto st = measure("athz transforms + rANS", run, n, true, 0x1FF, last && n == o->tableLogs.front(),
                                      n == o->tableLogs.front());
                    if (last && n == o->tableLogs.front()) transformed = st;
                    if (last && n == o->tableLogs.back()) transformed12 = st;
                }
            }
            out("\nEvery file decoded bit for bit (0 words differ, compared on the device).\n");
            // Per field of the splats' element (kind 1), summed over its planes.
            const uint32_t planeCount = static_cast<uint32_t>((*planes)[1].size());
            const std::vector<std::string> names = byteNames(*tree, planeCount);
            const uint32_t run0 = o->runs.back();
            struct Field {
                uint64_t raw = 0, r0 = 0, rt = 0, rt12 = 0, gz = 0, bytes = 0, ctxElem = 0, ctxByte = 0, res0 = 0;
                std::string layer;
                std::map<uint32_t, uint32_t> transforms;   // transform -> words
            };
            std::vector<std::pair<std::string, Field>> fields;
            const auto statOf = [](const lod::AthzStats& st, uint32_t kind, uint32_t b) -> const lod::AthzPlaneStat* {
                for (const lod::AthzPlaneStat& p : st.planes) {
                    if (p.kind == kind && p.byte == b) return &p;
                }
                return nullptr;
            };
            const auto cost = [](const lod::AthzPlaneStat* p) -> uint64_t {
                return p == nullptr ? 0 : (p->encoding == 1 ? p->ransBytes : p->rawBytes);
            };
            for (uint32_t b = 0; b < planeCount; ++b) {
                const std::string f = fieldOf(names[b]);
                if (fields.empty() || fields.back().first != f) fields.push_back({f, {}});
                Field& fd = fields.back().second;
                fd.bytes += tree->count;
                if (const auto* p = statOf(raw[run0], 1, b)) {
                    fd.raw += p->rawBytes;
                    fd.layer = std::string(1, "GATK"[p->layer]);
                }
                fd.r0 += cost(statOf(order0, 1, b));
                fd.rt12 += cost(statOf(transformed12, 1, b));
                if (const auto* p = statOf(transformed, 1, b)) {
                    fd.rt += cost(p);
                    fd.res0 += p->order0Bytes;
                    fd.ctxElem += std::min(p->contextElementBytes, p->order0Bytes);
                    fd.ctxByte += std::min(p->contextByteBytes, p->order0Bytes);
                    if (b % 4 == 0) ++fd.transforms[p->transform];
                }
                fd.gz += gzipPlane[1][b];
            }
            static const char* kTransformNames[] = {"none", "delta", "delta16", "xor", "parent", "parent16", "xorParent", "within", "parentWithin"};
            out("\nThe splats' planes by field (runs of %u, MB; rANS columns: the tables' entropy where a plane is coded "
                "so, its raw width where not; order 1: the residuals' entropy under each context, measured, not coded):\n\n",
                run0);
            out("| field | layer | as stored | raw | rANS order 0 | transforms + rANS 2^11 | 2^12 | words' transforms | "
                "order 1, previous element | order 1, previous byte | gzip of the planes |\n"
                "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|\n");
            uint64_t total[8] = {};
            for (const auto& [name, fd] : fields) {
                std::string ts;
                for (const auto& [t, n] : fd.transforms) {
                    ts += (ts.empty() ? "" : ", ") + std::string(kTransformNames[std::min(t, 8u)]) +
                          (fd.transforms.size() > 1 ? " " + std::to_string(n) : "");
                }
                out("| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %.2f | %.2f | %.2f |\n", name.c_str(),
                    fd.layer.c_str(), mb(fd.bytes), mb(fd.raw), mb(fd.r0), mb(fd.rt), mb(fd.rt12), ts.c_str(),
                    mb(fd.ctxElem), mb(fd.ctxByte), mb(fd.gz));
                const uint64_t v[8] = {fd.bytes, fd.raw, fd.r0, fd.rt, fd.rt12, fd.ctxElem, fd.ctxByte, fd.gz};
                for (int i = 0; i < 8; ++i) total[i] += v[i];
            }
            out("| all | | %.2f | %.2f | %.2f | %.2f | %.2f | | %.2f | %.2f | %.2f |\n", mb(total[0]), mb(total[1]),
                mb(total[2]), mb(total[3]), mb(total[4]), mb(total[5]), mb(total[6]), mb(total[7]));
            // Per plane, for the record (CSV beside the files).
            {
                const fs::path csv = dir / (stem + ".census.planes.csv");
                if (FILE* f = std::fopen(csv.string().c_str(), "w")) {
                    std::fprintf(f, "kind,byte,name,layer,width,raw_bytes,order0_enc,order0_bytes,transform,"
                                    "transform_enc,transform_bytes,residual_order0,order1_prev_element,"
                                    "order1_prev_byte,gzip_bytes\n");
                    for (uint32_t k = 0; k < 2; ++k) {
                        for (uint32_t b = 0; b < planeCount; ++b) {
                            const auto* pr = statOf(raw[run0], k, b);
                            const auto* p0 = statOf(order0, k, b);
                            const auto* pt = statOf(transformed, k, b);
                            std::fprintf(f, "%u,%u,%s,%u,%u,%llu,%u,%llu,%u,%u,%llu,%llu,%llu,%llu,%llu\n", k, b,
                                         names[b].c_str(), pr ? pr->layer : 0, pr ? pr->width : 0,
                                         static_cast<unsigned long long>(pr ? pr->rawBytes : 0), p0 ? p0->encoding : 0,
                                         static_cast<unsigned long long>(cost(p0)), pt ? pt->transform : 0,
                                         pt ? pt->encoding : 0, static_cast<unsigned long long>(cost(pt)),
                                         static_cast<unsigned long long>(pt ? pt->order0Bytes : 0),
                                         static_cast<unsigned long long>(pt ? pt->contextElementBytes : 0),
                                         static_cast<unsigned long long>(pt ? pt->contextByteBytes : 0),
                                         static_cast<unsigned long long>(gzipPlane[k][b]));
                        }
                    }
                    std::fclose(f);
                    out("\nPer plane: %s\n", csv.string().c_str());
                }
            }
            // The transfer as CPCA weights (E1b, 3): order 0 as they are and as a residual from the parent.
            if (!o->transferBasis.empty() && tree->splats.transferCount == 112) {
                auto cpca = cpcaCensus(library, *tree, o->transferBasis);
                if (!cpca) fail(cpca.error());
                out("\nThe transfer as CPCA weights (C 256, K 8, 4-bit weights; the splats; MB; lossy, against the "
                    "lossless transfer above): ids and weights at order 0 %.2f; as a residual from the parent group's "
                    "%.2f (ids agreeing with the parent's: %.1f %%).\n",
                    mb(cpca->raw), mb(cpca->residual), 100.0 * cpca->agree);
            }
            // Decode on the device, each batch three times, the least kept.
            out("\n| decode | runs | table | ms a page (median of the least of 3) | ms a batch (median of the least of 3) | "
                "batch MiB coded | GB/s decoded | clock |\n|---|---:|---:|---:|---:|---:|---:|---|\n");
            for (const auto& [coding, run, n, path] : timed) {
                auto timings = lod::timeAthzDecode(library, path, uint64_t{o->batchMiB} << 20, o->singlePages);
                if (!timings) fail(timings.error());
                std::vector<double> single, batch;
                double batchBytes = 0, decodedPerMs = 0;
                bool device = true;
                std::vector<lod::AthzDecodeTiming> full;
                for (const auto& t : *timings) {
                    device = device && t.fromDevice;
                    if (t.single) {
                        single.push_back(t.milliseconds);
                    } else {
                        full.push_back(t);
                    }
                }
                uint64_t most = 0;
                for (const auto& t : full) most = std::max(most, t.codedBytes);
                for (const auto& t : full) {
                    if (t.codedBytes * 10 >= most * 9 || full.size() == 1) {
                        batch.push_back(t.milliseconds);
                        batchBytes = static_cast<double>(t.codedBytes);
                        decodedPerMs = std::max(decodedPerMs, static_cast<double>(t.decodedBytes) / t.milliseconds);
                    }
                }
                out("| %s | %u | 2^%u | %.3f | %.3f | %.2f | %.1f | %s |\n", coding.c_str(), run, n, median(single),
                    median(batch), batchBytes / (1 << 20), decodedPerMs / 1.0e6, device ? "GPU" : "host");
                if (!o->keep) fs::remove(path, ec);
            }
            if (!o->keep) {
                fs::remove(athc, ec);
                fs::remove(planesPath, ec);
            }
        }
    });
}

}   // namespace athenea::cli
