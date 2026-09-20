#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

namespace fs = std::filesystem;

static const float BM25_K1 = 1.5f;
static const float BM25_B = 0.75f;

struct Doc {
    std::string id;
    std::string title;
    std::string text;
};

struct Query {
    std::string id;
    std::string text;
    std::vector<std::string> gold_ids;
};

struct Hit {
    int32_t doc;
    float score;
    int rank;
};

static void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

static std::string json_escape(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 8);
    o.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o.push_back(static_cast<char>(c));
                }
        }
    }
    o.push_back('"');
    return o;
}

static std::string cell_to_string(const std::shared_ptr<arrow::Array>& arr, int64_t i) {
    if (!arr || arr->IsNull(i)) return "";
    switch (arr->type_id()) {
        case arrow::Type::STRING:
            return std::string(std::static_pointer_cast<arrow::StringArray>(arr)->GetView(i));
        case arrow::Type::LARGE_STRING:
            return std::string(std::static_pointer_cast<arrow::LargeStringArray>(arr)->GetView(i));
        case arrow::Type::INT64:
            return std::to_string(std::static_pointer_cast<arrow::Int64Array>(arr)->Value(i));
        case arrow::Type::INT32:
            return std::to_string(std::static_pointer_cast<arrow::Int32Array>(arr)->Value(i));
        case arrow::Type::DICTIONARY: {
            auto dict = std::static_pointer_cast<arrow::DictionaryArray>(arr);
            return cell_to_string(dict->dictionary(), dict->GetValueIndex(i));
        }
        default:
            return arr->GetScalar(i).ValueOrDie()->ToString();
    }
}

static std::shared_ptr<arrow::Array> col(const std::shared_ptr<arrow::RecordBatch>& batch,
                                         std::initializer_list<const char*> names) {
    for (const char* name : names) {
        auto c = batch->GetColumnByName(name);
        if (c) return c;
    }
    return nullptr;
}

static std::vector<fs::path> arrow_shards(const fs::path& dir) {
    std::vector<fs::path> out;
    if (!fs::exists(dir)) die("missing " + dir.string());
    for (const auto& entry : fs::directory_iterator(dir)) {
        auto name = entry.path().filename().string();
        if (name.rfind("data-", 0) == 0 && entry.path().extension() == ".arrow") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) die("no arrow shards in " + dir.string());
    return out;
}

static void for_each_batch(const fs::path& dir,
                           const std::function<void(const std::shared_ptr<arrow::RecordBatch>&)>& fn) {
    for (const auto& path : arrow_shards(dir)) {
        auto mm = arrow::io::MemoryMappedFile::Open(path.string(), arrow::io::FileMode::READ);
        if (!mm.ok()) die("mmap failed: " + path.string() + " " + mm.status().ToString());
        auto reader = arrow::ipc::RecordBatchStreamReader::Open(*mm);
        if (!reader.ok()) die("arrow open failed: " + path.string() + " " + reader.status().ToString());
        while (true) {
            auto batch = (*reader)->Next();
            if (!batch.ok()) die(batch.status().ToString());
            if (!*batch) break;
            fn(*batch);
        }
    }
}

static void tokenize(std::string_view text, std::vector<std::string>& tokens) {
    tokens.clear();
    std::string tok;
    tok.reserve(16);
    for (unsigned char c : text) {
        if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c + 32);
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            tok.push_back(static_cast<char>(c));
        } else if (!tok.empty()) {
            tokens.push_back(tok);
            tok.clear();
        }
    }
    if (!tok.empty()) tokens.push_back(tok);
}

class BM25Index {
public:
    std::vector<Doc> docs;
    float avgdl = 0.f;

    void build() {
        const int32_t n = static_cast<int32_t>(docs.size());
        doc_len.assign(n, 0.f);
        vocab.clear();
        postings.clear();
        idf.clear();
        vocab.reserve(1 << 18);

        std::vector<std::string> tokens;
        std::unordered_map<int32_t, int32_t> tf;
        double len_sum = 0.0;

        for (int32_t i = 0; i < n; ++i) {
            tokens.clear();
            tokenize(docs[i].title, tokens);
            tokenize(docs[i].text, tokens);
            doc_len[i] = static_cast<float>(tokens.size());
            len_sum += doc_len[i];

            tf.clear();
            for (const auto& t : tokens) {
                auto [it, inserted] = vocab.emplace(t, static_cast<int32_t>(vocab.size()));
                if (inserted) postings.emplace_back();
                tf[it->second] += 1;
            }
            for (auto [tid, freq] : tf) {
                postings[tid].push_back({i, static_cast<uint16_t>(std::min(freq, 65535))});
            }
        }
        avgdl = n ? static_cast<float>(len_sum / n) : 0.f;
        idf.resize(postings.size());
        for (size_t t = 0; t < postings.size(); ++t) {
            const float df = static_cast<float>(postings[t].size());
            idf[t] = std::log(1.f + (static_cast<float>(n) - df + 0.5f) / (df + 0.5f));
        }
    }

    void retrieve(std::string_view query, int top_k, std::vector<Hit>& hits,
                  std::vector<float>& scores, std::vector<int32_t>& touched) const {
        hits.clear();
        for (int32_t d : touched) scores[d] = 0.f;
        touched.clear();
        if (docs.empty() || top_k <= 0) return;

        std::vector<std::string> tokens;
        tokenize(query, tokens);
        std::sort(tokens.begin(), tokens.end());
        tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());

        const float k1 = BM25_K1;
        const float b = BM25_B;
        const float avg = avgdl > 0.f ? avgdl : 1.f;

        for (const auto& tok : tokens) {
            auto it = vocab.find(tok);
            if (it == vocab.end()) continue;
            const int32_t tid = it->second;
            const float idf_t = idf[tid];
            for (auto [doc, tf] : postings[tid]) {
                const float f = static_cast<float>(tf);
                const float denom = f + k1 * (1.f - b + b * doc_len[doc] / avg);
                if (scores[doc] == 0.f) touched.push_back(doc);
                scores[doc] += idf_t * (f * (k1 + 1.f)) / denom;
            }
        }

        int k = std::min(top_k, static_cast<int>(touched.size()));
        if (k <= 0) return;
        std::partial_sort(
            touched.begin(), touched.begin() + k, touched.end(),
            [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });
        hits.reserve(k);
        for (int i = 0; i < k; ++i) {
            const int32_t d = touched[i];
            if (scores[d] <= 0.f) break;
            hits.push_back({d, scores[d], i + 1});
        }
    }

private:
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::vector<std::pair<int32_t, uint16_t>>> postings;
    std::vector<float> idf;
    std::vector<float> doc_len;
};

static std::string pack_doc(const Doc& doc, const Hit* hit) {
    std::ostringstream o;
    o << "{\"_id\":" << json_escape(doc.id)
      << ",\"title\":" << json_escape(doc.title)
      << ",\"text\":" << json_escape(doc.text);
    if (hit) {
        o << ",\"bm25_rank\":" << hit->rank
          << ",\"bm25_score\":" << hit->score;
    }
    o << "}";
    return o.str();
}

static std::unordered_map<std::string, std::vector<std::string>> load_qrels(const fs::path& dir) {
    std::unordered_map<std::string, std::vector<std::string>> qrels;
    for_each_batch(dir, [&](const auto& batch) {
        auto qid = col(batch, {"query-id", "query_id"});
        auto did = col(batch, {"corpus-id", "corpus_id"});
        auto score = col(batch, {"score"});
        if (!qid || !did) die("qrels missing columns");
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            if (score) {
                const std::string s = cell_to_string(score, i);
                if (s == "0" || s.empty()) continue;
            }
            qrels[cell_to_string(qid, i)].push_back(cell_to_string(did, i));
        }
    });
    return qrels;
}

static std::unordered_map<std::string, std::string> load_queries(const fs::path& dir) {
    std::unordered_map<std::string, std::string> queries;
    for_each_batch(dir, [&](const auto& batch) {
        auto id = col(batch, {"_id"});
        auto text = col(batch, {"text"});
        if (!id || !text) die("queries missing columns");
        queries.reserve(queries.size() + static_cast<size_t>(batch->num_rows()));
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            queries.emplace(cell_to_string(id, i), cell_to_string(text, i));
        }
    });
    return queries;
}

static std::vector<Doc> load_corpus_pool(const fs::path& dir,
                                         const std::unordered_set<std::string>& gold,
                                         int extra_docs, bool full, uint32_t seed) {
    std::vector<Doc> pool;
    std::vector<Doc> extras;
    extras.reserve(static_cast<size_t>(std::max(extra_docs, 0)));
    int64_t seen_non_gold = 0;
    std::mt19937 rng(seed);

    for_each_batch(dir, [&](const auto& batch) {
        auto id = col(batch, {"_id"});
        auto title = col(batch, {"title"});
        auto text = col(batch, {"text"});
        if (!id || !text) die("corpus missing columns");
        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            Doc doc;
            doc.id = cell_to_string(id, i);
            doc.title = title ? cell_to_string(title, i) : "";
            doc.text = cell_to_string(text, i);
            if (full) {
                pool.push_back(std::move(doc));
                continue;
            }
            if (gold.count(doc.id)) {
                pool.push_back(std::move(doc));
                continue;
            }
            if (extra_docs <= 0) continue;
            ++seen_non_gold;
            if (static_cast<int>(extras.size()) < extra_docs) {
                extras.push_back(std::move(doc));
            } else {
                std::uniform_int_distribution<int64_t> dist(0, seen_non_gold - 1);
                const int64_t j = dist(rng);
                if (j < extra_docs) extras[static_cast<size_t>(j)] = std::move(doc);
            }
        }
    });
    pool.insert(pool.end(), std::make_move_iterator(extras.begin()),
                std::make_move_iterator(extras.end()));
    return pool;
}

struct Args {
    fs::path data_dir;
    std::string split = "train";
    fs::path output;
    int top_k = 10;
    int max_queries = -1;
    int extra_docs = 0;
    bool full_corpus = false;
    uint32_t seed = 42;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
};

static Args parse_args(int argc, char** argv) {
    Args a;
    const fs::path exe = fs::absolute(argv[0]);
    fs::path root = exe.parent_path();
    if (root.filename() == "training") root = root.parent_path();
    a.data_dir = root / "data" / "hotpotqa";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--data-dir") a.data_dir = need("--data-dir");
        else if (arg == "--split") a.split = need("--split");
        else if (arg == "--output") a.output = need("--output");
        else if (arg == "--top-k") a.top_k = std::stoi(need("--top-k"));
        else if (arg == "--max-queries") a.max_queries = std::stoi(need("--max-queries"));
        else if (arg == "--extra-docs") a.extra_docs = std::stoi(need("--extra-docs"));
        else if (arg == "--seed") a.seed = static_cast<uint32_t>(std::stoul(need("--seed")));
        else if (arg == "--threads") a.threads = std::max(1, std::stoi(need("--threads")));
        else if (arg == "--full-corpus") a.full_corpus = true;
        else die("unknown arg: " + arg);
    }
    if (a.split != "train" && a.split != "validation" && a.split != "test") {
        die("split must be train|validation|test");
    }
    if (a.output.empty()) a.output = a.data_dir / "hard_negatives" / (a.split + ".jsonl");
    return a;
}

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    Args args = parse_args(argc, argv);

    std::cerr << "loading qrels ...\n";
    auto qrels_map = load_qrels(args.data_dir / "qrels" / args.split);
    std::vector<std::string> qids;
    qids.reserve(qrels_map.size());
    for (const auto& [qid, _] : qrels_map) qids.push_back(qid);
    std::sort(qids.begin(), qids.end());
    if (args.max_queries > 0 && args.max_queries < static_cast<int>(qids.size())) {
        std::mt19937 rng(args.seed);
        std::shuffle(qids.begin(), qids.end(), rng);
        qids.resize(static_cast<size_t>(args.max_queries));
    }

    std::unordered_set<std::string> gold_ids;
    std::vector<Query> queries;
    queries.reserve(qids.size());
    for (const auto& qid : qids) {
        Query q;
        q.id = qid;
        q.gold_ids = qrels_map[qid];
        for (const auto& did : q.gold_ids) gold_ids.insert(did);
        queries.push_back(std::move(q));
    }

    std::cerr << "loading queries ...\n";
    auto query_text = load_queries(args.data_dir / "queries");
    for (auto& q : queries) {
        auto it = query_text.find(q.id);
        if (it == query_text.end()) die("missing query text for " + q.id);
        q.text = it->second;
    }

    std::cerr << "loading corpus ...\n";
    BM25Index index;
    index.docs = load_corpus_pool(args.data_dir / "corpus", gold_ids, args.extra_docs,
                                  args.full_corpus, args.seed);
    std::unordered_map<std::string, int32_t> doc_pos;
    doc_pos.reserve(index.docs.size() * 2);
    for (int32_t i = 0; i < static_cast<int32_t>(index.docs.size()); ++i) {
        doc_pos.emplace(index.docs[i].id, i);
    }

    std::cerr << "pool=" << index.docs.size() << " queries=" << queries.size()
              << " gold_docs=" << gold_ids.size() << "\nindexing BM25 ...\n";
    index.build();

    fs::create_directories(args.output.parent_path());
    const int threads = std::min(args.threads, std::max(1, static_cast<int>(queries.size())));
    std::vector<std::string> lines(queries.size());
    std::atomic<int> n_written{0};
    std::atomic<int> n_short{0};
    std::atomic<size_t> next{0};

    auto worker = [&]() {
        std::vector<float> scores(index.docs.size(), 0.f);
        std::vector<int32_t> touched;
        std::vector<Hit> hits;
        std::unordered_set<std::string> gold;
        while (true) {
            const size_t i = next.fetch_add(1);
            if (i >= queries.size()) break;
            const Query& q = queries[i];
            gold.clear();
            gold.insert(q.gold_ids.begin(), q.gold_ids.end());

            std::string positives;
            int n_pos = 0;
            for (const auto& did : q.gold_ids) {
                auto it = doc_pos.find(did);
                if (it == doc_pos.end()) continue;
                if (n_pos) positives += ",";
                positives += pack_doc(index.docs[it->second], nullptr);
                ++n_pos;
            }
            if (!n_pos) continue;

            index.retrieve(q.text, args.top_k + n_pos + 5, hits, scores, touched);
            std::string negatives;
            int n_neg = 0;
            for (const auto& hit : hits) {
                const Doc& doc = index.docs[hit.doc];
                if (gold.count(doc.id)) continue;
                if (n_neg) negatives += ",";
                negatives += pack_doc(doc, &hit);
                if (++n_neg >= args.top_k) break;
            }
            if (n_neg < args.top_k) n_short.fetch_add(1);

            lines[i] = std::string("{\"query_id\":") + json_escape(q.id) + ",\"query\":" +
                       json_escape(q.text) + ",\"positives\":[" + positives +
                       "],\"hard_negatives\":[" + negatives + "]}\n";
            n_written.fetch_add(1);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    std::ofstream out(args.output, std::ios::binary);
    if (!out) die("cannot write " + args.output.string());
    for (const auto& line : lines) {
        if (!line.empty()) out << line;
    }

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    std::cerr << "wrote " << n_written.load() << " records to " << args.output.string() << "\n";
    std::cerr << "queries with fewer than " << args.top_k << " hard negatives: " << n_short.load()
              << "\n";
    std::cerr << "elapsed_ms=" << ms << " threads=" << threads << "\n";
    return 0;
}
