#include "loam_baseline/keypoint_extractor.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "loam_baseline/pca.hpp"

namespace {

// Returns indices of v sorted in descending order.
std::vector<std::size_t> sortIdx(const std::vector<double>& v) {
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::ranges::sort(idx, [&v](auto a, auto b) { return v[a] > v[b]; });
  return idx;
}

class LineFitting {
 public:
  // Fit a line through pts by PCA; returns true if all pts are within
  // max_distance_ of the fitted line.
  bool fitPCA(std::vector<Eigen::Vector3d>& pts);

  // First checks that successive segment directions are consistent
  // (sin of angle < sin(max_angle_)), then calls fitPCA.
  bool fitPCAAndCheckConsistency(std::vector<Eigen::Vector3d>& pts);

  double squaredDistanceToPoint(const Eigen::Vector3d& pt) const;

  Eigen::Vector3d direction_;
  Eigen::Vector3d position_;
  double max_distance_ = 0.02;
  double max_angle_ = 40.0 * std::numbers::pi_v<double> / 180.0;
};

bool LineFitting::fitPCA(std::vector<Eigen::Vector3d>& pts) {
  const auto [eig, mean] = loam_baseline::ComputePCA(pts);
  position_ = mean;
  direction_ = eig.eigenvectors().col(2).normalized();

  const double sq_max = max_distance_ * max_distance_;
  for (const auto& p : pts) {
    if (squaredDistanceToPoint(p) > sq_max) {
      return false;
    }
  }
  return true;
}

bool LineFitting::fitPCAAndCheckConsistency(std::vector<Eigen::Vector3d>& pts) {
  const double max_sin = std::sin(max_angle_);
  // u is the direction of the first segment; compare each subsequent segment
  // direction against it.
  const Eigen::Vector3d u = (pts[1] - pts[0]).normalized();
  for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
    const Eigen::Vector3d v = (pts[i + 1] - pts[i]).normalized();
    if ((u.cross(v)).norm() > max_sin) {
      return false;
    }
  }
  return fitPCA(pts);
}

double LineFitting::squaredDistanceToPoint(const Eigen::Vector3d& pt) const {
  return ((pt - position_).cross(direction_)).squaredNorm();
}

}  // namespace

namespace loam_baseline {

KeypointExtractor::KeypointExtractor(KeypointExtractorCfg cfg) : cfg(cfg) {}

KeypointExtractorResult KeypointExtractor::computeKeyPoints(
    const pcl::PointCloud<Point>& cloud, int n_scans, bool dynamic_mask) {
  n_lasers_ = n_scans;
  current_frame_ = std::make_shared<pcl::PointCloud<Point>>(cloud);

  prepareForNextFrame();
  convertAndSortScanLines();

  // Initialise per-point arrays for each scan line in parallel.
  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, static_cast<std::size_t>(n_lasers_)),
      [this](const tbb::blocked_range<std::size_t>& r) {
        for (std::size_t s = r.begin(); s != r.end(); ++s) {
          const std::size_t n = frame_by_scan_[s]->size();
          is_point_valid_[s].assign(n, KeypointFlags().set());
          label_[s].assign(n, KeypointFlags());
          angles_[s].assign(n, 0.0);
          saliency_[s].assign(n, 0.0);
          depth_gap_[s].assign(n, 0.0);
          intensity_gap_[s].assign(n, 0.0);
        }
      });

  invalidPointWithBadCriteria(dynamic_mask);
  computeCurvature();

  KeypointExtractorResult out;
  setKeyPointsLabels(out);
  return out;
}

// ---------------------------------------------------------------------------

void KeypointExtractor::prepareForNextFrame() {
  const auto n = static_cast<std::size_t>(n_lasers_);
  frame_by_scan_.resize(n);
  for (auto& scan : frame_by_scan_) {
    if (scan) {
      scan->clear();
    } else {
      scan = std::make_shared<pcl::PointCloud<Point>>();
    }
  }

  angles_.resize(n);
  saliency_.resize(n);
  depth_gap_.resize(n);
  intensity_gap_.resize(n);
  is_point_valid_.resize(n);
  label_.resize(n);
}

void KeypointExtractor::convertAndSortScanLines() {
  const std::size_t n_pts = current_frame_->size();
  const double t_start = static_cast<double>(current_frame_->points[0].time);
  const double t_end =
      static_cast<double>(current_frame_->points[n_pts - 1].time);

  for (std::size_t i = 0; i < n_pts; ++i) {
    Point pt = current_frame_->points[i];
    // Normalise time to [0, 1] within the frame.
    pt.time =
        static_cast<float>((static_cast<double>(pt.time) - t_start) / t_end);
    const std::size_t id = static_cast<std::size_t>(pt.laserId);
    frame_by_scan_[id]->push_back(pt);
  }
}

void KeypointExtractor::computeCurvature() {
  const double sq_dist_line =
      cfg.dist_to_line_threshold * cfg.dist_to_line_threshold;
  constexpr double kSqDepthCoeff = 0.25;
  constexpr double kMinDepthGap = 1.5;  // [m]

  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, static_cast<std::size_t>(n_lasers_)),
      [&](const tbb::blocked_range<std::size_t>& r) {
        // Per-thread scratch buffers — allocated once per task range.
        const std::size_t w = static_cast<std::size_t>(cfg.neighbor_width);
        std::vector<Eigen::Vector3d> left_nb(w);
        std::vector<Eigen::Vector3d> right_nb(w);
        std::vector<Eigen::Vector3d> far_nb;
        far_nb.reserve(2 * w);

        LineFitting left_line;
        LineFitting right_line;
        LineFitting far_line;
        left_line.max_distance_ = cfg.dist_to_line_threshold;
        right_line.max_distance_ = cfg.dist_to_line_threshold;
        far_line.max_distance_ = cfg.dist_to_line_threshold;

        for (std::size_t s = r.begin(); s != r.end(); ++s) {
          const int n_pts = static_cast<int>(frame_by_scan_[s]->size());
          if (isScanLineAlmostEmpty(n_pts)) {
            continue;
          }

          for (int idx = cfg.neighbor_width; idx + cfg.neighbor_width < n_pts;
               ++idx) {
            const std::size_t uidx = static_cast<std::size_t>(idx);
            if (is_point_valid_[s][uidx].none()) {
              continue;
            }

            const Point& cp = frame_by_scan_[s]->points[uidx];
            const Eigen::Vector3d central(cp.x, cp.y, cp.z);

            intensity_gap_[s][uidx] =
                std::abs(static_cast<double>(
                             frame_by_scan_[s]->points[uidx + 1].intensity) -
                         static_cast<double>(
                             frame_by_scan_[s]->points[uidx - 1].intensity));

            for (int j = idx - 1; j >= idx - cfg.neighbor_width; --j) {
              const Point& p =
                  frame_by_scan_[s]->points[static_cast<std::size_t>(j)];
              left_nb[static_cast<std::size_t>(idx - 1 - j)] << p.x, p.y, p.z;
            }
            for (int j = idx + 1; j <= idx + cfg.neighbor_width; ++j) {
              const Point& p =
                  frame_by_scan_[s]->points[static_cast<std::size_t>(j)];
              right_nb[static_cast<std::size_t>(j - idx - 1)] << p.x, p.y, p.z;
            }

            const bool left_flat = left_line.fitPCAAndCheckConsistency(left_nb);
            const bool right_flat =
                right_line.fitPCAAndCheckConsistency(right_nb);

            double dist_left = 0.0;
            double dist_right = 0.0;

            if (left_flat && right_flat) {
              dist_left = left_line.squaredDistanceToPoint(central);
              dist_right = right_line.squaredDistanceToPoint(central);
              if (dist_left < sq_dist_line && dist_right < sq_dist_line) {
                angles_[s][uidx] =
                    (left_line.direction_.cross(right_line.direction_)).norm();
              }
            } else if (!left_flat && right_flat) {
              dist_left = std::numeric_limits<double>::max();
              for (const auto& ln : left_nb) {
                dist_left =
                    std::min(dist_left, right_line.squaredDistanceToPoint(ln));
              }
              dist_left *= kSqDepthCoeff;
            } else if (left_flat && !right_flat) {
              dist_right = std::numeric_limits<double>::max();
              for (const auto& rn : right_nb) {
                dist_right =
                    std::min(dist_right, left_line.squaredDistanceToPoint(rn));
              }
              dist_right *= kSqDepthCoeff;
            } else {
              // Neither neighbourhood is flat: look for a depth gap.
              const double curr_depth = central.norm();
              bool left_gap = false;
              bool right_gap = false;
              far_nb.clear();

              for (const auto& ln : left_nb) {
                if (std::abs(ln.norm() - curr_depth) > kMinDepthGap) {
                  left_gap = true;
                  far_nb.push_back(ln);
                } else if (left_gap) {
                  break;
                }
              }
              for (const auto& rn : right_nb) {
                if (std::abs(rn.norm() - curr_depth) > kMinDepthGap) {
                  right_gap = true;
                  far_nb.push_back(rn);
                } else if (right_gap) {
                  break;
                }
              }

              if (far_nb.size() > w) {
                far_line.fitPCA(far_nb);
                saliency_[s][uidx] = far_line.squaredDistanceToPoint(central);
              }
            }

            depth_gap_[s][uidx] = std::max(dist_left, dist_right);
          }
        }
      });
}

void KeypointExtractor::invalidPointWithBadCriteria(bool dynamic_mask) {
  constexpr double kExpectedCoeff = 10.0;

  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, static_cast<std::size_t>(n_lasers_)),
      [&](const tbb::blocked_range<std::size_t>& r) {
        for (std::size_t s = r.begin(); s != r.end(); ++s) {
          const int n_pts = static_cast<int>(frame_by_scan_[s]->size());

          if (isScanLineAlmostEmpty(n_pts)) {
            for (int i = 0; i < n_pts; ++i) {
              is_point_valid_[s][static_cast<std::size_t>(i)].reset();
            }
            continue;
          }

          // Border points lack a full neighbourhood.
          for (int i = 0; i <= cfg.neighbor_width; ++i) {
            is_point_valid_[s][static_cast<std::size_t>(i)].reset();
          }
          for (int i = n_pts - 1 - cfg.neighbor_width - 1; i < n_pts; ++i) {
            is_point_valid_[s][static_cast<std::size_t>(i)].reset();
          }

          for (int idx = cfg.neighbor_width;
               idx < n_pts - cfg.neighbor_width - 1; ++idx) {
            const std::size_t uidx = static_cast<std::size_t>(idx);
            const Point& pp = frame_by_scan_[s]->points[uidx - 1];
            const Point& cp = frame_by_scan_[s]->points[uidx];
            const Point& np = frame_by_scan_[s]->points[uidx + 1];

            const Eigen::Vector3f xp = pp.getVector3fMap();
            const Eigen::Vector3f x = cp.getVector3fMap();
            const Eigen::Vector3f xn = np.getVector3fMap();

            const double depth = static_cast<double>(x.norm());
            const double depth_next = static_cast<double>(xn.norm());
            const double d_next = static_cast<double>((xn - x).norm());
            const double d_prev = static_cast<double>((x - xp).norm());
            const double expected = cfg.angle_resolution * depth;

            if (dynamic_mask) {
              constexpr double kMaxZ = 0.4;
              constexpr double kMinZ = -0.8;
              constexpr double kMaxX = 10.0;
              constexpr double kMaxAngle = std::numbers::pi_v<double> / 12.0;
              const double azimuth = std::atan2(static_cast<double>(cp.y),
                                                static_cast<double>(cp.x));
              if (cp.z < kMaxZ && cp.z > kMinZ && cp.x < kMaxX &&
                  std::abs(azimuth) < kMaxAngle) {
                is_point_valid_[s][uidx].reset();
              }
              if (cp.z < kMaxZ && cp.z > kMinZ && cp.x > -kMaxX &&
                  (azimuth < -std::numbers::pi_v<double> + kMaxAngle ||
                   azimuth > std::numbers::pi_v<double> - kMaxAngle)) {
                is_point_valid_[s][uidx].reset();
              }
            }

            if (d_next > kExpectedCoeff * expected) {
              if (depth < depth_next) {
                is_point_valid_[s][uidx + 1].reset();
                for (int i = idx + 2; i <= idx + cfg.neighbor_width; ++i) {
                  const Eigen::Vector3f y =
                      frame_by_scan_[s]
                          ->points[static_cast<std::size_t>(i) - 1]
                          .getVector3fMap();
                  const Eigen::Vector3f yn =
                      frame_by_scan_[s]
                          ->points[static_cast<std::size_t>(i)]
                          .getVector3fMap();
                  if ((yn - y).norm() > kExpectedCoeff * expected) {
                    break;
                  }
                  is_point_valid_[s][static_cast<std::size_t>(i)].reset();
                }
              } else {
                is_point_valid_[s][uidx].reset();
                for (int i = idx - cfg.neighbor_width; i < idx; ++i) {
                  const Eigen::Vector3f yp =
                      frame_by_scan_[s]
                          ->points[static_cast<std::size_t>(i)]
                          .getVector3fMap();
                  const Eigen::Vector3f y =
                      frame_by_scan_[s]->points[i + 1].getVector3fMap();
                  if ((y - yp).norm() > kExpectedCoeff * expected) {
                    break;
                  }
                  is_point_valid_[s][static_cast<std::size_t>(i)].reset();
                }
              }
            }

            if (depth < cfg.min_distance_to_sensor) {
              is_point_valid_[s][uidx].reset();
            }

            // Invalidate if too far, or if the point sits in an isolated gap
            // (neighbour spacing far exceeds the expected azimuthal step).
            if (depth > cfg.max_distance_to_sensor ||
                (d_prev > 0.25 * kExpectedCoeff * expected &&
                 d_next > 0.25 * kExpectedCoeff * expected)) {
              is_point_valid_[s][uidx].reset();
            }
          }
        }
      });
}

void KeypointExtractor::setKeyPointsLabels(KeypointExtractorResult& out) {
  const double sq_saliency =
      cfg.edge_saliency_threshold * cfg.edge_saliency_threshold;
  const double sq_depth_gap =
      cfg.edge_depth_gap_threshold * cfg.edge_depth_gap_threshold;

  tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, static_cast<std::size_t>(n_lasers_)),
      [&](const tbb::blocked_range<std::size_t>& r) {
        for (std::size_t s = r.begin(); s != r.end(); ++s) {
          const int n_pts = static_cast<int>(frame_by_scan_[s]->size());
          if (isScanLineAlmostEmpty(n_pts)) {
            continue;
          }

          const auto depth_idx = sortIdx(depth_gap_[s]);
          const auto angle_idx = sortIdx(angles_[s]);
          const auto saliency_idx = sortIdx(saliency_[s]);
          const auto intensity_idx = sortIdx(intensity_gap_[s]);

          // Mark edges by sorted criterion; invalidate neighbours of each
          // selected edge so they cannot also be selected.
          const auto add_edges = [&](const std::vector<std::size_t>& sorted,
                                     const std::vector<double>& values,
                                     double threshold, int radius) {
            for (const std::size_t i : sorted) {
              if (values[i] < threshold) {
                break;
              }
              if (!is_point_valid_[s][i][kEdge]) {
                continue;
              }
              label_[s][i].set(kEdge);
              const int begin = std::max(0, static_cast<int>(i) - radius);
              const int end = std::min(n_pts - 1, static_cast<int>(i) + radius);
              for (int j = begin; j <= end; ++j) {
                is_point_valid_[s][static_cast<std::size_t>(j)].reset(kEdge);
              }
            }
          };

          add_edges(depth_idx, depth_gap_[s], sq_depth_gap,
                    cfg.neighbor_width - 1);
          add_edges(angle_idx, angles_[s], cfg.edge_sin_angle_threshold,
                    cfg.neighbor_width);
          add_edges(saliency_idx, saliency_[s], sq_saliency,
                    cfg.neighbor_width - 1);
          add_edges(intensity_idx, intensity_gap_[s],
                    cfg.edge_intensity_gap_threshold, 1);

          // Planes: iterate from the smallest sin-angle upward.
          for (int k = n_pts - 1; k >= 0; --k) {
            const std::size_t i = angle_idx[static_cast<std::size_t>(k)];
            if (angles_[s][i] > cfg.plane_sin_angle_threshold) {
              break;
            }
            if (!is_point_valid_[s][i][kPlane]) {
              continue;
            }
            label_[s][i].set(kPlane);
            const int begin = std::max(0, static_cast<int>(i) - 4);
            const int end = std::min(n_pts - 1, static_cast<int>(i) + 4);
            for (int j = begin; j <= end; ++j) {
              is_point_valid_[s][static_cast<std::size_t>(j)].reset(kPlane);
            }
          }

          // Blobs: every 3rd still-valid point.
          for (int i = 0; i < n_pts; i += 3) {
            if (is_point_valid_[s][static_cast<std::size_t>(i)][kBlob]) {
              label_[s][static_cast<std::size_t>(i)].set(kBlob);
            }
          }
        }
      });

  // Collect into result (serial to avoid concurrent_vector overhead).
  for (std::size_t s = 0; s < static_cast<std::size_t>(n_lasers_); ++s) {
    for (std::size_t i = 0; i < frame_by_scan_[s]->size(); ++i) {
      const Point& p = frame_by_scan_[s]->points[i];
      if (label_[s][i][kEdge]) {
        out.edges.push_back(p);
      }
      if (label_[s][i][kPlane]) {
        out.planes.push_back(p);
      }
    }
  }
}

}  // namespace loam_baseline
