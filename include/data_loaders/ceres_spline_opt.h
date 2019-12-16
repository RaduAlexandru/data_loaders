#ifndef CERES_SPLINE_OPT_H_
#define CERES_SPLINE_OPT_H_

#include <unordered_map>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include<Eigen/StdVector>

class SplineImpl;
typedef std::shared_ptr<SplineImpl> SplineImplPtr;

class CeresSplineOptimization
{
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
   
   bool getPose ( const double & tcur, Eigen::Affine3d & pose_wb );
   bool isTimeValid ( const double & t_query ) const;
   bool calculatePoseSpline ( const double & t_cur );
   void initSpline ( const std::vector<std::pair<uint64_t, Eigen::Affine3d>,  Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Affine3d>>  > & poses_w_b );

   double m_min_ts = 0;
private:
   SplineImplPtr m_spline;
   std::vector<std::pair<double, Eigen::Affine3d>,  Eigen::aligned_allocator<std::pair<double, Eigen::Affine3d>>  > m_vec_poses_wb;
};

typedef std::shared_ptr<CeresSplineOptimization> CeresSplineOptimizationPtr;

#endif // CERES_SPLINE_OPT_H_
