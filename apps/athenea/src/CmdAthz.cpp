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
            std::map<uint32_t, lod::AthzStats> raw, rans11, rans12;
            std::map<std::pair<uint32_t, uint32_t>, fs::path> ransPaths;
            for (const uint32_t run : o->runs) {
                lod::AthzSettings s;
                s.runGaussians = run;
                s.pageRuns = 3072 / run;
                s.rans = false;
                const fs::path rawPath = dir / (stem + ".census.raw" + std::to_string(run) + ".athz");
                auto r = lod::writeAthz(library, *tree, rawPath, s);
                if (!r) fail(r.error());
                raw[run] = *r;
                row("athz raw bit-pack", run, "-", r->fileBytes);
                if (!o->keep) fs::remove(rawPath, ec);
                for (const uint32_t n : o->tableLogs) {
                    s.rans = true;
                    s.tableLog = n;
                    const fs::path path = dir / (stem + ".census.r" + std::to_string(run) + "n" + std::to_string(n) + ".athz");
                    auto st = lod::writeAthz(library, *tree, path, s);
                    if (!st) fail(st.error());
                    (n == 11 ? rans11 : rans12)[run] = *st;
                    ransPaths[{run, n}] = path;
                    row("athz rANS order 0", run, "2^" + std::to_string(n), st->fileBytes);
                    // E1.3: the decoded record is the record, word for word.
                    auto back = lod::readAthz(library, path);
                    if (!back) fail(back.error());
                    auto differ = recordDifference(library, *tree, *back);
                    if (!differ) fail(differ.error());
                    if (*differ != 0) {
                        fail(Error::make(ErrorCode::InternalError, "{}: {} words decoded differently", path.string(),
                                         *differ));
                    }
                }
            }
            out("\nEvery rANS file decoded bit for bit (0 words differ, compared on the device).\n");
            // Per field of the splats' element (kind 1), summed over its planes.
            const uint32_t planeCount = static_cast<uint32_t>((*planes)[1].size());
            const std::vector<std::string> names = byteNames(*tree, planeCount);
            const uint32_t run0 = o->runs.back();
            struct Field {
                uint64_t raw = 0, r11 = 0, r12 = 0, gz = 0, bytes = 0;
                std::string layer;
                uint32_t rans11 = 0;
            };
            std::vector<std::pair<std::string, Field>> fields;
            const auto statOf = [](const lod::AthzStats& st, uint32_t kind, uint32_t b) -> const lod::AthzPlaneStat* {
                for (const lod::AthzPlaneStat& p : st.planes) {
                    if (p.kind == kind && p.byte == b) return &p;
                }
                return nullptr;
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
                if (rans11.count(run0) != 0) {
                    if (const auto* p = statOf(rans11[run0], 1, b)) {
                        fd.r11 += p->encoding == 1 ? p->ransBytes : p->rawBytes;
                        fd.rans11 += p->encoding;
                    }
                }
                if (rans12.count(run0) != 0) {
                    if (const auto* p = statOf(rans12[run0], 1, b)) fd.r12 += p->encoding == 1 ? p->ransBytes : p->rawBytes;
                }
                fd.gz += gzipPlane[1][b];
            }
            out("\nThe splats' planes by field (runs of %u; rANS columns: the tables' entropy where a plane is coded "
                "so, its raw width where not; MB):\n\n",
                run0);
            out("| field | layer | as stored | raw width | rANS 2^11 | planes rANS | rANS 2^12 | gzip of the planes |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|\n");
            for (const auto& [name, fd] : fields) {
                out("| %s | %s | %.2f | %.2f | %.2f | %u | %.2f | %.2f |\n", name.c_str(), fd.layer.c_str(),
                    mb(fd.bytes), mb(fd.raw), mb(fd.r11), fd.rans11, mb(fd.r12), mb(fd.gz));
            }
            // Per plane, for the record (CSV beside the files).
            {
                const fs::path csv = dir / (stem + ".census.planes.csv");
                if (FILE* f = std::fopen(csv.string().c_str(), "w")) {
                    std::fprintf(f, "kind,byte,name,layer,width,raw_bytes,rans11_enc,rans11_bytes,rans12_bytes,gzip_bytes\n");
                    for (uint32_t k = 0; k < 2; ++k) {
                        for (uint32_t b = 0; b < planeCount; ++b) {
                            const auto* pr = statOf(raw[run0], k, b);
                            const auto* p11 = rans11.count(run0) ? statOf(rans11[run0], k, b) : nullptr;
                            const auto* p12 = rans12.count(run0) ? statOf(rans12[run0], k, b) : nullptr;
                            std::fprintf(f, "%u,%u,%s,%u,%u,%llu,%u,%llu,%llu,%llu\n", k, b, names[b].c_str(),
                                         pr ? pr->layer : 0, pr ? pr->width : 0,
                                         static_cast<unsigned long long>(pr ? pr->rawBytes : 0), p11 ? p11->encoding : 0,
                                         static_cast<unsigned long long>(p11 ? p11->ransBytes : 0),
                                         static_cast<unsigned long long>(p12 ? p12->ransBytes : 0),
                                         static_cast<unsigned long long>(gzipPlane[k][b]));
                        }
                    }
                    std::fclose(f);
                    out("\nPer plane: %s\n", csv.string().c_str());
                }
            }
            // Decode on the device.
            out("\n| decode | runs | table | ms a page (median) | ms a batch (median) | batch MiB coded | "
                "GB/s decoded | clock |\n|---|---:|---:|---:|---:|---:|---:|---|\n");
            for (const auto& [key, path] : ransPaths) {
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
                // The batches that are full (the last holds what was left).
                uint64_t most = 0;
                for (const auto& t : full) most = std::max(most, t.codedBytes);
                for (const auto& t : full) {
                    if (t.codedBytes * 10 >= most * 9 || full.size() == 1) {
                        batch.push_back(t.milliseconds);
                        batchBytes = static_cast<double>(t.codedBytes);
                        decodedPerMs = std::max(decodedPerMs, static_cast<double>(t.decodedBytes) / t.milliseconds);
                    }
                }
                out("| athz rANS | %u | 2^%u | %.3f | %.3f | %.2f | %.1f | %s |\n", key.first, key.second,
                    median(single), median(batch), batchBytes / (1 << 20), decodedPerMs / 1.0e6,
                    device ? "GPU" : "host");
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
