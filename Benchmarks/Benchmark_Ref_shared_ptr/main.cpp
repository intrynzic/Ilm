// ============================================================================
// Test-RefBenchmark: ilm::Ref vs std::shared_ptr
//
// Compares ilm::Ref against std::shared_ptr created two ways:
//   std_make_shared  std::make_shared<T>()      (1 allocation: object + control block)
//   std_shared_new   std::shared_ptr<T>(new T)  (2 allocations: same layout as ilm::CreateRef)
//   ilm_ref          ilm::CreateRef<T>()
//
// Output (in --out-dir, default ./bench_results), all meant for pandas/matplotlib:
//   timings.csv  tidy/long format, one row per (benchmark, impl, param, repeat)
//   memory.csv   allocation counts and bytes per object, plus handle sizes
//   meta.json    compiler, build type, CPU thread count, run settings
//
// Build (GCC/Clang, from this folder):
//   g++ -std=c++23 -O3 -DNDEBUG -I../../Ilm/src/include main.cpp -o ref_bench -pthread
// Run:
//   ./ref_bench --out-dir results
//   python3 plot.py results
//
// For stable numbers: Release build, plugged in, and ideally pinned to a few
// cores (e.g. `taskset -c 2-5 ./ref_bench`) with the CPU governor on "performance".
// ============================================================================

#include <Ilm/Ilm.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__GLIBC__)
#   include <malloc.h>
#endif
#if defined(_MSC_VER)
#   include <intrin.h>
#endif

// ----------------------------------------------------------------------------
// Optimizer barriers
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
// Forces `value` to exist in memory and be considered read/modified by opaque code.
template<typename T>
inline void DoNotOptimize(const T& value) noexcept { asm volatile("" : : "r"(&value) : "memory"); }
inline void ClobberMemory() noexcept { asm volatile("" : : : "memory"); }
#elif defined(_MSC_VER)
static const void* volatile g_Sink;
template<typename T>
inline void DoNotOptimize(const T& value) noexcept { g_Sink = &value; _ReadWriteBarrier(); }
inline void ClobberMemory() noexcept { _ReadWriteBarrier(); }
#else
#   error "Unsupported compiler: add DoNotOptimize/ClobberMemory"
#endif

// ----------------------------------------------------------------------------
// Allocation tracking (global operator new/delete replacement).
// Only counts while Enabled is true, which is only the case in the memory
// phase, so it costs the timed benchmarks a single predictable branch.
// ----------------------------------------------------------------------------
namespace AllocTracker
{
    inline bool Enabled = false;
    inline uint64_t Count = 0;
    inline uint64_t Bytes = 0;
    inline uint64_t Usable = 0;

#if defined(__GLIBC__)
    inline constexpr bool HasUsable = true;
#else
    inline constexpr bool HasUsable = false;
#endif

    inline void Reset() { Count = Bytes = Usable = 0; }
}

// noinline: matches the real operator new/delete (which live in the C++ runtime and are never
// inlined), and stops GCC from pairing inlined new/delete with malloc/free diagnostics.
#if defined(_MSC_VER)
#   define BENCH_NOINLINE __declspec(noinline)
#else
#   define BENCH_NOINLINE __attribute__((noinline))
#endif

BENCH_NOINLINE void* operator new(std::size_t size)
{
    void* p = std::malloc(size ? size : 1);
    if (!p)
        throw std::bad_alloc();

    if (AllocTracker::Enabled)
    {
        ++AllocTracker::Count;
        AllocTracker::Bytes += size;
#if defined(__GLIBC__)
        AllocTracker::Usable += malloc_usable_size(p);
#endif
    }

    return p;
}

BENCH_NOINLINE void operator delete(void* p) noexcept { std::free(p); }
BENCH_NOINLINE void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace
{
    using Clock = std::chrono::steady_clock;

    inline double ElapsedNs(Clock::time_point a, Clock::time_point b) noexcept
    {
        return std::chrono::duration<double, std::nano>(b - a).count();
    }

    double Median(std::vector<double> v)
    {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    // ----------------------------------------------------------------------------
    // Payloads
    // ----------------------------------------------------------------------------
    // User-provided constructor => default-initialized storage (no memset), so every
    // implementation does identical work constructing the object.
    template<size_t N>
    struct Payload
    {
        Payload() noexcept { data[0] = 1; }
        unsigned char data[N];
    };

    struct Base
    {
        virtual ~Base() = default;
        virtual int F() const noexcept { return 1; }
        int b = 0;
    };

    struct Derived : Base
    {
        int F() const noexcept override { return 2; }
        int d = 0;
    };

    // ----------------------------------------------------------------------------
    // Implementation adapters: give every smart pointer the same tiny interface
    // ----------------------------------------------------------------------------
    struct StdMakeShared
    {
        static constexpr const char* Key = "std_make_shared";
        template<class T> using Ptr = std::shared_ptr<T>;
        template<class T> using Weak = std::weak_ptr<T>;

        template<class T, class... A> static Ptr<T> Create(A&&... a) { return std::make_shared<T>(std::forward<A>(a)...); }
        template<class T> static Weak<T> MakeWeak(const Ptr<T>& p) { return Weak<T>(p); }
        template<class T> static Ptr<T> Lock(const Weak<T>& w) { return w.lock(); }
        template<class T> static void Reset(Ptr<T>& p) { p.reset(); }
    };

    struct StdSharedNew
    {
        static constexpr const char* Key = "std_shared_new";
        template<class T> using Ptr = std::shared_ptr<T>;
        template<class T> using Weak = std::weak_ptr<T>;

        template<class T, class... A> static Ptr<T> Create(A&&... a) { return Ptr<T>(new T(std::forward<A>(a)...)); }
        template<class T> static Weak<T> MakeWeak(const Ptr<T>& p) { return Weak<T>(p); }
        template<class T> static Ptr<T> Lock(const Weak<T>& w) { return w.lock(); }
        template<class T> static void Reset(Ptr<T>& p) { p.reset(); }
    };

    struct IlmRef
    {
        static constexpr const char* Key = "ilm_ref";
        template<class T> using Ptr = ilm::Ref<T>;
        template<class T> using Weak = ilm::WeakRef<T>;

        template<class T, class... A> static Ptr<T> Create(A&&... a) { return ilm::CreateRef<T>(std::forward<A>(a)...); }
        template<class T> static Weak<T> MakeWeak(const Ptr<T>& p) { return Weak<T>(p); }
        template<class T> static Ptr<T> Lock(const Weak<T>& w) { return w.Lock(); }
        template<class T> static void Reset(Ptr<T>& p) { p.Reset(); }
    };

    template<class T> struct Tag { using type = T; };

    template<class F>
    void ForImpls(F&& f)
    {
        f(Tag<StdMakeShared>{});   // first one is the baseline for the console ratios
        f(Tag<StdSharedNew>{});
        f(Tag<IlmRef>{});
    }

    // ----------------------------------------------------------------------------
    // Runner
    // ----------------------------------------------------------------------------
    struct Config
    {
        int repeats = 15;
        double minTimeMs = 20.0;
        unsigned maxThreads = 0;     // 0 = auto
        std::string filter;
        std::string outDir = "bench_results";
        bool allowSingleThreadFastPath = false;
    };

    struct Case
    {
        std::string impl;
        double itemsPerIter;                       // operations performed per iteration
        std::function<double(uint64_t)> run;       // runs `iters` iterations, returns total ns
    };

    class Runner
    {
    public:
        explicit Runner(Config cfg) : m_Cfg(std::move(cfg)) {}

        const Config& Cfg() const { return m_Cfg; }

        bool Enabled(const char* name) const
        {
            return m_Cfg.filter.empty() || std::string(name).find(m_Cfg.filter) != std::string::npos;
        }

        bool Open()
        {
            std::error_code ec;
            std::filesystem::create_directories(m_Cfg.outDir, ec);

            m_Csv.open(std::filesystem::path(m_Cfg.outDir) / "timings.csv");
            if (!m_Csv)
                return false;

            m_Csv << "benchmark,impl,param_name,param,threads,repeat,iterations,ns_per_op\n";
            return true;
        }

        void Run(const char* name, const char* paramName, int64_t param, unsigned threads, std::vector<Case>& cases)
        {
            // Calibrate iteration counts (this also serves as warmup)
            std::vector<uint64_t> iters;
            for (auto& c : cases)
                iters.push_back(Calibrate(c));

            // Interleave implementations across repeats so thermal/frequency drift
            // affects all of them equally; rotate the starting impl each repeat.
            std::vector<std::vector<double>> samples(cases.size());
            const size_t n = cases.size();
            for (int r = 0; r < m_Cfg.repeats; ++r)
            {
                for (size_t k = 0; k < n; ++k)
                {
                    const size_t i = (k + static_cast<size_t>(r)) % n;
                    const double totalNs = cases[i].run(iters[i]);
                    const double perOp = totalNs / (static_cast<double>(iters[i]) * cases[i].itemsPerIter);
                    samples[i].push_back(perOp);

                    char line[512];
                    std::snprintf(line, sizeof(line), "%s,%s,%s,%lld,%u,%d,%llu,%.6f\n",
                        name, cases[i].impl.c_str(), paramName, static_cast<long long>(param), threads, r,
                        static_cast<unsigned long long>(iters[i]), perOp);
                    m_Csv << line;
                }
            }
            m_Csv.flush();

            // Console summary
            char label[96];
            if (std::strcmp(paramName, "none") == 0)
                std::snprintf(label, sizeof(label), "%s", name);
            else
                std::snprintf(label, sizeof(label), "%s[%s=%lld]", name, paramName, static_cast<long long>(param));

            const double base = Median(samples[0]);
            for (size_t i = 0; i < n; ++i)
            {
                const double med = Median(samples[i]);
                std::fprintf(stderr, "  %-44s %-16s %10.3f ns/op   x%.2f\n",
                    label, cases[i].impl.c_str(), med, base > 0 ? med / base : 0.0);
            }
            std::fprintf(stderr, "\n");
        }

    private:
        uint64_t Calibrate(Case& c) const
        {
            const double target = m_Cfg.minTimeMs * 1e6;
            uint64_t iters = 1;
            for (;;)
            {
                const double t = c.run(iters);
                if (t >= target || iters >= (1ull << 32))
                    return iters;

                double scale = t > 0 ? (target * 1.1 / t) : 10.0;
                scale = std::clamp(scale, 1.5, 10.0);
                iters = std::max<uint64_t>(iters + 1, static_cast<uint64_t>(static_cast<double>(iters) * scale));
            }
        }

    private:
        Config m_Cfg;
        std::ofstream m_Csv;
    };

    // ----------------------------------------------------------------------------
    // Benchmarks: single-threaded
    // ----------------------------------------------------------------------------

    // Create + destroy one object (steady state: allocator free lists stay hot).
    template<size_t N>
    void BenchCreateDestroy(Runner& R)
    {
        if (!R.Enabled("create_destroy")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<N>;
                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    auto p = I::template Create<T>();
                    DoNotOptimize(p);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("create_destroy", "object_size", N, 1, cases);
    }

    constexpr size_t kBatch = 4096;

    // Create a batch of objects, then destroy them all (allocator is not just recycling one slot).
    // ns/op is per object.
    template<size_t N>
    void BenchCreateBatch(Runner& R)
    {
        if (!R.Enabled("create_batch")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, static_cast<double>(kBatch), [](uint64_t iters) -> double {
                using T = Payload<N>;
                using P = typename I::template Ptr<T>;
                std::vector<P> v;
                v.reserve(kBatch);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    for (size_t j = 0; j < kBatch; ++j)
                        v.push_back(I::template Create<T>());

                    DoNotOptimize(v);
                    v.clear();
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("create_batch", "object_size", N, 1, cases);
    }

    // Copy-construct a handle and destroy it (refcount inc + dec).
    void BenchCopyDestroy(Runner& R)
    {
        if (!R.Enabled("copy_destroy")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                auto p = I::template Create<T>();
                DoNotOptimize(p);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    auto c = p;
                    DoNotOptimize(c);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("copy_destroy", "none", 0, 1, cases);
    }

    // Copy-assign onto a handle that already owns something else (inc new, dec old). ns/op per assignment.
    void BenchCopyAssign(Runner& R)
    {
        if (!R.Enabled("copy_assign")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 2.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                auto s1 = I::template Create<T>();
                auto s2 = I::template Create<T>();
                auto cur = s1;
                DoNotOptimize(s1);
                DoNotOptimize(s2);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    cur = s2;
                    DoNotOptimize(cur);
                    cur = s1;
                    DoNotOptimize(cur);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("copy_assign", "none", 0, 1, cases);
    }

    // Move-construct then move-assign back (no refcount traffic expected; measures handle size/shuffling).
    // One op = one move-construct + one move-assign.
    void BenchMoveRoundtrip(Runner& R)
    {
        if (!R.Enabled("move_roundtrip")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                using P = typename I::template Ptr<T>;
                P a = I::template Create<T>();
                DoNotOptimize(a);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    P b(std::move(a));
                    DoNotOptimize(b);
                    a = std::move(b);
                    DoNotOptimize(a);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("move_roundtrip", "none", 0, 1, cases);
    }

    // operator-> plus a read of the pointee.
    void BenchDeref(Runner& R)
    {
        if (!R.Enabled("deref")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                auto p = I::template Create<T>();
                DoNotOptimize(p);
                uint64_t sum = 0;

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    sum += p->data[0];
                    ClobberMemory();
                }
                const double ns = ElapsedNs(t0, Clock::now());
                DoNotOptimize(sum);
                return ns;
            } });
            });

        R.Run("deref", "none", 0, 1, cases);
    }

    // Implicit Derived -> Base conversion copy.
    void BenchUpcastCopy(Runner& R)
    {
        if (!R.Enabled("upcast_copy")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                auto d = I::template Create<Derived>();
                DoNotOptimize(d);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    typename I::template Ptr<Base> b = d;
                    DoNotOptimize(b);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("upcast_copy", "none", 0, 1, cases);
    }

    // Construct a weak handle from a strong one and destroy it.
    void BenchWeakCreate(Runner& R)
    {
        if (!R.Enabled("weak_create")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                auto p = I::template Create<T>();
                DoNotOptimize(p);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    auto w = I::template MakeWeak<T>(p);
                    DoNotOptimize(w);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("weak_create", "none", 0, 1, cases);
    }

    // Lock a weak handle whose object is alive, then release the strong handle.
    void BenchWeakLock(Runner& R)
    {
        if (!R.Enabled("weak_lock")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                auto p = I::template Create<T>();
                auto w = I::template MakeWeak<T>(p);
                DoNotOptimize(p);
                DoNotOptimize(w);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    auto l = I::template Lock<T>(w);
                    DoNotOptimize(l);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("weak_lock", "none", 0, 1, cases);
    }

    // Lock a weak handle whose object has already been destroyed (should return null quickly).
    void BenchWeakLockExpired(Runner& R)
    {
        if (!R.Enabled("weak_lock_expired")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [](uint64_t iters) -> double {
                using T = Payload<64>;
                auto p = I::template Create<T>();
                auto w = I::template MakeWeak<T>(p);
                I::template Reset<T>(p);
                DoNotOptimize(w);

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    auto l = I::template Lock<T>(w);
                    DoNotOptimize(l);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("weak_lock_expired", "none", 0, 1, cases);
    }

    // ----------------------------------------------------------------------------
    // Benchmarks: containers of distinct objects (cache behaviour). ns/op is per element.
    // ----------------------------------------------------------------------------

    // Copy a vector of N handles to N distinct objects (refcount blocks scattered on the heap).
    void BenchContainerCopy(Runner& R, size_t n)
    {
        if (!R.Enabled("container_copy")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            using T = Payload<64>;
            using P = typename I::template Ptr<T>;

            auto v = std::make_shared<std::vector<P>>();
            v->reserve(n);
            for (size_t i = 0; i < n; ++i)
                v->push_back(I::template Create<T>());

            cases.push_back(Case{ I::Key, static_cast<double>(n), [v](uint64_t iters) -> double {
                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    std::vector<P> copy = *v;
                    DoNotOptimize(copy);
                }
                return ElapsedNs(t0, Clock::now());
            } });
            });

        R.Run("container_copy", "container_size", static_cast<int64_t>(n), 1, cases);
    }

    // Walk a shuffled vector of N handles and read each pointee (pointer chasing across the heap).
    void BenchContainerIterate(Runner& R, size_t n)
    {
        if (!R.Enabled("container_iterate")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            using T = Payload<64>;
            using P = typename I::template Ptr<T>;

            auto v = std::make_shared<std::vector<P>>();
            v->reserve(n);
            for (size_t i = 0; i < n; ++i)
                v->push_back(I::template Create<T>());

            std::mt19937_64 rng(12345);
            std::shuffle(v->begin(), v->end(), rng);

            cases.push_back(Case{ I::Key, static_cast<double>(n), [v](uint64_t iters) -> double {
                uint64_t sum = 0;
                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    for (const auto& p : *v)
                        sum += p->data[0];

                    ClobberMemory();
                }
                const double ns = ElapsedNs(t0, Clock::now());
                DoNotOptimize(sum);
                return ns;
            } });
            });

        R.Run("container_iterate", "container_size", static_cast<int64_t>(n), 1, cases);
    }

    // ----------------------------------------------------------------------------
    // Benchmarks: multi-threaded
    // ----------------------------------------------------------------------------

    // All threads copy+destroy handles to the SAME object (contended refcount cache line).
    // ns/op is the mean per-thread latency of one copy+destroy; plot.py derives throughput.
    template<class I>
    double ContendedCopy(unsigned threads, uint64_t iters)
    {
        using T = Payload<64>;
        using P = typename I::template Ptr<T>;

        P shared = I::template Create<T>();
        std::atomic<unsigned> ready{ 0 };
        std::atomic<bool> go{ false };
        std::vector<double> elapsed(threads, 0.0);
        std::vector<std::thread> pool;
        pool.reserve(threads);

        for (unsigned t = 0; t < threads; ++t)
        {
            pool.emplace_back([&, t] {
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (!go.load(std::memory_order_acquire))
                    std::this_thread::yield();

                const auto t0 = Clock::now();
                for (uint64_t i = 0; i < iters; ++i)
                {
                    P c = shared;
                    DoNotOptimize(c);
                }
                elapsed[t] = ElapsedNs(t0, Clock::now());
                });
        }

        while (ready.load(std::memory_order_acquire) != threads)
            std::this_thread::yield();

        go.store(true, std::memory_order_release);
        for (auto& th : pool)
            th.join();

        return std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / static_cast<double>(threads);
    }

    void BenchContendedCopy(Runner& R, unsigned threads)
    {
        if (!R.Enabled("contended_copy")) return;

        std::vector<Case> cases;
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            cases.push_back(Case{ I::Key, 1.0, [threads](uint64_t iters) -> double {
                return ContendedCopy<I>(threads, iters);
            } });
            });

        R.Run("contended_copy", "threads", threads, threads, cases);
    }

    // ----------------------------------------------------------------------------
    // Memory footprint (allocation count and bytes per object)
    // ----------------------------------------------------------------------------
    template<size_t N>
    void MeasureMemoryFor(std::ofstream& csv)
    {
        ForImpls([&](auto tag) {
            using I = typename decltype(tag)::type;
            using T = Payload<N>;
            using P = typename I::template Ptr<T>;

            AllocTracker::Reset();
            AllocTracker::Enabled = true;
            P p = I::template Create<T>();
            AllocTracker::Enabled = false;
            DoNotOptimize(p);

            csv << I::Key << ',' << N << ',' << sizeof(P) << ',' << AllocTracker::Count << ',' << AllocTracker::Bytes << ',';
            if (AllocTracker::HasUsable)
                csv << AllocTracker::Usable;
            csv << '\n';
            });
    }

    void MeasureMemory(const Config& cfg)
    {
        std::ofstream csv(std::filesystem::path(cfg.outDir) / "memory.csv");
        csv << "impl,object_size,sizeof_ptr,allocations,bytes_requested,bytes_usable\n";

        MeasureMemoryFor<8>(csv);
        MeasureMemoryFor<32>(csv);
        MeasureMemoryFor<64>(csv);
        MeasureMemoryFor<128>(csv);
        MeasureMemoryFor<256>(csv);
        MeasureMemoryFor<1024>(csv);
        MeasureMemoryFor<4096>(csv);

        std::fprintf(stderr, "Memory footprint written to %s\n\n", (std::filesystem::path(cfg.outDir) / "memory.csv").string().c_str());
    }

    // ----------------------------------------------------------------------------
    // Metadata
    // ----------------------------------------------------------------------------
    std::string JsonEscape(const std::string& s)
    {
        std::string out;
        for (const char c : s)
        {
            if (c == '"' || c == '\\') { out += '\\'; out += c; }
            else if (c == '\n') out += "\\n";
            else out += c;
        }
        return out;
    }

    std::vector<unsigned> ThreadCounts(const Config& cfg)
    {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned maxT = cfg.maxThreads ? cfg.maxThreads : std::min(hw * 2, 32u);

        std::vector<unsigned> v;
        for (unsigned t = 1; t <= maxT; t *= 2)
            v.push_back(t);

        if (hw <= maxT && std::find(v.begin(), v.end(), hw) == v.end())
            v.push_back(hw);

        std::sort(v.begin(), v.end());
        return v;
    }

    void WriteMeta(const Config& cfg, const std::vector<unsigned>& threadCounts)
    {
        std::string compiler;
#if defined(__clang__)
        compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        compiler = std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
        compiler = "msvc " + std::to_string(_MSC_FULL_VER);
#endif

        const char* os =
#if defined(_WIN32)
            "windows";
#elif defined(__APPLE__)
            "macos";
#elif defined(__linux__)
            "linux";
#else
            "unknown";
#endif

        std::time_t now = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        gmtime_s(&tm, &now);
#else
        gmtime_r(&now, &tm);
#endif
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", &tm);

#if defined(NDEBUG)
        const bool optimized = true;
#else
        const bool optimized = false;
#endif

        std::ofstream f(std::filesystem::path(cfg.outDir) / "meta.json");
        f << "{\n";
        f << "  \"timestamp_utc\": \"" << stamp << "\",\n";
        f << "  \"compiler\": \"" << JsonEscape(compiler) << "\",\n";
        f << "  \"cplusplus\": " << __cplusplus << ",\n";
        f << "  \"ndebug\": " << (optimized ? "true" : "false") << ",\n";
        f << "  \"os\": \"" << os << "\",\n";
        f << "  \"pointer_bits\": " << sizeof(void*) * 8 << ",\n";
        f << "  \"hardware_threads\": " << std::thread::hardware_concurrency() << ",\n";
        f << "  \"single_thread_fast_path_allowed\": " << (cfg.allowSingleThreadFastPath ? "true" : "false") << ",\n";
        f << "  \"repeats\": " << cfg.repeats << ",\n";
        f << "  \"min_time_ms\": " << cfg.minTimeMs << ",\n";
        f << "  \"thread_counts\": [";
        for (size_t i = 0; i < threadCounts.size(); ++i)
            f << (i ? ", " : "") << threadCounts[i];
        f << "],\n";
        f << "  \"sizeof\": { \"ilm_ref\": " << sizeof(ilm::Ref<int>)
            << ", \"std_shared_ptr\": " << sizeof(std::shared_ptr<int>) << " }\n";
        f << "}\n";
    }

    void PrintUsage()
    {
        std::fprintf(stderr,
            "ilm::Ref vs std::shared_ptr benchmark\n\n"
            "  --out-dir <dir>       output directory (default: bench_results)\n"
            "  --repeats <n>         samples per benchmark/impl (default: 15)\n"
            "  --min-time-ms <ms>    target time per sample (default: 20)\n"
            "  --filter <substr>     only run benchmarks whose name contains <substr>\n"
            "  --max-threads <n>     highest thread count for contended_copy (default: 2x hw threads, max 32)\n"
            "  --allow-single-thread-fast-path\n"
            "                        do NOT spawn a warm-up thread. libstdc++ then uses NON-atomic refcounts\n"
            "                        in std::shared_ptr until the first thread is created, which is unfair to\n"
            "                        ilm::Ref (always atomic) and unrepresentative of multithreaded programs\n"
            "  --quick               fast smoke run (5 repeats, 5 ms samples)\n"
            "  --help\n");
    }
} // namespace

int main(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "Missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
            };

        if (a == "--out-dir") cfg.outDir = next("--out-dir");
        else if (a == "--repeats") cfg.repeats = std::max(1, std::atoi(next("--repeats")));
        else if (a == "--min-time-ms") cfg.minTimeMs = std::max(0.1, std::atof(next("--min-time-ms")));
        else if (a == "--filter") cfg.filter = next("--filter");
        else if (a == "--max-threads") cfg.maxThreads = static_cast<unsigned>(std::max(1, std::atoi(next("--max-threads"))));
        else if (a == "--allow-single-thread-fast-path") cfg.allowSingleThreadFastPath = true;
        else if (a == "--quick") { cfg.repeats = 5; cfg.minTimeMs = 5.0; }
        else if (a == "--help" || a == "-h") { PrintUsage(); return 0; }
        else { std::fprintf(stderr, "Unknown argument: %s\n\n", a.c_str()); PrintUsage(); return 2; }
    }

#if !defined(NDEBUG)
    std::fprintf(stderr, "WARNING: NDEBUG is not defined. These numbers are only meaningful for an optimized (Release) build.\n\n");
#endif

    // libstdc++'s shared_ptr skips atomic refcount operations while the process is
    // single-threaded (__libc_single_threaded), and that flag never flips back once a
    // thread has been created. ilm::Ref is always atomic, so by default we spawn and
    // join a throwaway thread up front so BOTH use atomics for every benchmark, which is
    // what a real multithreaded program (e.g. a game engine) gets.
    if (!cfg.allowSingleThreadFastPath)
        std::thread([] {}).join();

    Runner runner(cfg);
    if (!runner.Open())
    {
        std::fprintf(stderr, "Could not open output directory '%s'\n", cfg.outDir.c_str());
        return 1;
    }

    const auto threadCounts = ThreadCounts(cfg);
    WriteMeta(cfg, threadCounts);

    std::fprintf(stderr, "ilm::Ref vs std::shared_ptr | repeats=%d min-time=%.1fms | sizeof(ilm::Ref)=%zu sizeof(shared_ptr)=%zu\n",
        cfg.repeats, cfg.minTimeMs, sizeof(ilm::Ref<int>), sizeof(std::shared_ptr<int>));
    std::fprintf(stderr, "std::shared_ptr single-thread fast path: %s\n\n",
        cfg.allowSingleThreadFastPath ? "ALLOWED (non-atomic refcounts possible)" : "defeated (atomics everywhere)");

    MeasureMemory(cfg);

    // Object lifetime cost
    BenchCreateDestroy<8>(runner);
    BenchCreateDestroy<32>(runner);
    BenchCreateDestroy<64>(runner);
    BenchCreateDestroy<128>(runner);
    BenchCreateDestroy<256>(runner);
    BenchCreateDestroy<1024>(runner);
    BenchCreateDestroy<4096>(runner);
    BenchCreateBatch<16>(runner);
    BenchCreateBatch<64>(runner);
    BenchCreateBatch<256>(runner);

    // Handle operations
    BenchCopyDestroy(runner);
    BenchCopyAssign(runner);
    BenchMoveRoundtrip(runner);
    BenchDeref(runner);
    BenchUpcastCopy(runner);
    BenchWeakCreate(runner);
    BenchWeakLock(runner);
    BenchWeakLockExpired(runner);

    // Containers (cache effects)
    for (const size_t n : { 16u, 128u, 1024u, 8192u, 65536u, 262144u })
        BenchContainerCopy(runner, n);

    for (const size_t n : { 16u, 128u, 1024u, 8192u, 65536u, 262144u })
        BenchContainerIterate(runner, n);

    // Contention
    for (const unsigned t : threadCounts)
        BenchContendedCopy(runner, t);

    std::fprintf(stderr, "Done. Results in %s/ (timings.csv, memory.csv, meta.json)\n", cfg.outDir.c_str());
    return 0;
}
