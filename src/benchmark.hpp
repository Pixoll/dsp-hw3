#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

inline constexpr int PRECISION = 9;

struct PhaseSample {
    double phase1 = 0.0;
    double phase2 = 0.0;
    double phase3 = 0.0;

    [[nodiscard]] double total() const {
        return phase1 + phase2 + phase3;
    }
};

struct Stats {
    double mean = -1;
    double stdev = -1;
    std::array<double, 5> quartiles{-1, -1, -1, -1, -1};

    void calculate(std::vector<double> values) {
        if (values.empty()) {
            return;
        }

        const auto count = static_cast<double>(values.size());

        mean = 0;
        for (const double v: values) {
            mean += v;
        }
        mean /= count;

        stdev = 0;
        if (values.size() > 1) {
            for (const double v: values) {
                const double dev = v - mean;
                stdev += dev * dev;
            }
            stdev /= count - 1;
            stdev = std::sqrt(stdev);
        }

        std::ranges::sort(values);
        const size_t sz = values.size();

        quartiles[0] = values.front();
        quartiles[4] = values.back();

        if (sz % 2 == 1) {
            quartiles[2] = values[sz / 2];
        } else {
            const size_t part = sz / 2;
            quartiles[2] = (values[part - 1] + values[part]) / 2.0;
        }

        if (sz % 4 >= 2) {
            quartiles[1] = values[sz / 4];
            quartiles[3] = values[3 * sz / 4];
        } else {
            size_t part = sz / 4;
            quartiles[1] = part > 0 ? 0.25 * values[part - 1] + 0.75 * values[part] : values[0];
            part = 3 * sz / 4;
            quartiles[3] = 0.75 * values[part - 1] + 0.25 * values[part];
        }
    }
};

struct Result {
    int experiment;
    int m = -1;
    int n = -1;
    int width = -1;
    int height = -1;
    int streams = -1;
    int samples = 0;

    Stats phase1, phase2, phase3, total;

    double speedup = -1;
    double efficiency = -1;

    bool has_correct = false;
    bool correct = false;

    [[nodiscard]] bool timed() const {
        return total.mean != -1;
    }
};

inline constexpr auto CSV_HEADER =
    "experiment,m,n,width,height,streams,samples,"
    "phase1_mean,phase1_stdev,"
    "phase2_mean,phase2_stdev,"
    "phase3_mean,phase3_stdev,"
    "total_mean,total_stdev,total_q0,total_q1,total_q2,total_q3,total_q4,"
    "speedup,efficiency,correct";

inline std::ofstream &operator<<(std::ofstream &out, const Result &r) {
    out << r.experiment << ','
        << r.m << ','
        << r.n << ','
        << r.width << ','
        << r.height << ',';

    if (r.streams != -1) out << r.streams;
    out << ',';

    out << r.samples << ','
        << r.phase1.mean << ',' << r.phase1.stdev << ','
        << r.phase2.mean << ',' << r.phase2.stdev << ','
        << r.phase3.mean << ',' << r.phase3.stdev << ','
        << r.total.mean << ',' << r.total.stdev << ','
        << r.total.quartiles[0] << ',' << r.total.quartiles[1] << ','
        << r.total.quartiles[2] << ',' << r.total.quartiles[3] << ','
        << r.total.quartiles[4] << ',';

    if (r.speedup != -1) out << r.speedup;
    out << ',';
    if (r.efficiency != -1) out << r.efficiency;
    out << ',';

    if (r.has_correct) out << std::boolalpha << r.correct;

    out << '\n';

    return out;
}

inline std::ofstream open_measurements_csv(const std::filesystem::path &path) {
    if (path.has_parent_path() && !std::filesystem::exists(path.parent_path())) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream out(path, std::ios::trunc);
    out << std::fixed << std::setprecision(PRECISION);
    out << CSV_HEADER << '\n';
    return out;
}

class Benchmark {
    int m_experiment;
    int m_m;
    int m_n;
    int m_width;
    int m_height;
    int m_streams;
    std::vector<PhaseSample> m_samples;
    bool m_has_correct = false;
    bool m_correct = false;

public:
    Benchmark(
        const int experiment,
        const int m,
        const int n,
        const int width,
        const int height,
        const int streams = -1
    ) : m_experiment(experiment), m_m(m), m_n(n), m_width(width), m_height(height), m_streams(streams) {
    }

    void add_sample(const PhaseSample &s) {
        m_samples.push_back(s);
    }

    void set_correct(const bool correct) {
        m_has_correct = true;
        m_correct = correct;
    }

    [[nodiscard]] Result finalize(const double baseline_total_mean = -1) const {
        Result result = {
            .experiment = m_experiment,
            .m = m_m,
            .n = m_n,
            .width = m_width,
            .height = m_height,
            .streams = m_streams,
            .samples = static_cast<int>(m_samples.size()),
            .has_correct = m_has_correct,
            .correct = m_correct,
        };

        std::vector<double> p1;
        std::vector<double> p2;
        std::vector<double> p3;
        std::vector<double> total;
        p1.reserve(m_samples.size());
        p2.reserve(m_samples.size());
        p3.reserve(m_samples.size());
        total.reserve(m_samples.size());

        for (const auto &s: m_samples) {
            p1.push_back(s.phase1);
            p2.push_back(s.phase2);
            p3.push_back(s.phase3);
            total.push_back(s.total());
        }

        result.phase1.calculate(p1);
        result.phase2.calculate(p2);
        result.phase3.calculate(p3);
        result.total.calculate(total);

        if (m_streams > 1 && baseline_total_mean > 0 && result.timed()) {
            result.speedup = baseline_total_mean / result.total.mean;
            result.efficiency = result.speedup / m_streams;
        }

        return result;
    }
};
