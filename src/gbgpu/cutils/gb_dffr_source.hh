#ifndef __GB_DFFR_SOURCE_HH__
#define __GB_DFFR_SOURCE_HH__

// The galactic-binary source model of the DFFR response: the contract of lat_dffr_kernels.hh for
// the nine parameters (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta) of GBTDIonTheFly, with
// one component. The phase and amplitude come from gb_physics.hh, which GBTDIonTheFly evaluates
// too, and the strain is that of LISATDIonTheFly::get_hp_hc:
//   h+ = -A (1 + cos^2 iota) cos(Phi),  hx = -2 A cos(iota) sin(Phi),  rotated by 2 psi.

#include "gb_physics.hh"
#include "lat_dffr_kernels.hh"

#include <cmath>

struct GBDFFRSource {
  static constexpr int num_components = 1;
  static constexpr int num_params = 9;

  // The launch inputs of the model: the epoch of the parameters, in the time frame of the orbits.
  struct Model {
    double t_ref;
  };

  struct Constants {
    DFFRSourceGeometry geometry;
    double amplitude;
    double frequency;
    double frequency_derivative;
    double frequency_second_derivative;
    double initial_phase;
    double plus_weight;
    double cross_weight;
    double time_reference;
  };

  CUDA_DEVICE static void load(Constants *source, const Model &model, const double *params) {
    source->amplitude = params[0];
    source->frequency = params[1];
    source->frequency_derivative = params[2];
    source->frequency_second_derivative = params[3];
    source->initial_phase = params[4];
    dffr_fill_source_geometry(&source->geometry, params[6], params[7], params[8]);
    // * LAT's strain factor -2 folds into the two polarisation weights; it cancels in the
    // * envelope slope, which is a logarithmic derivative.
    const double cos_inclination = std::cos(params[5]);
    source->plus_weight = DFFR_LAT_STRAIN_FACTOR * 0.5 * (1.0 + cos_inclination * cos_inclination);
    source->cross_weight = DFFR_LAT_STRAIN_FACTOR * cos_inclination;
    source->time_reference = model.t_ref;
  }

  CUDA_DEVICE static double time_reference(const Constants &source) {
    return source.time_reference;
  }

  CUDA_DEVICE static double amplitude(const Constants &source) { return source.amplitude; }

  CUDA_DEVICE static double plus_weight(const Constants &source, int component) {
    (void)component;
    return source.plus_weight;
  }

  CUDA_DEVICE static double cross_weight(const Constants &source, int component) {
    (void)component;
    return source.cross_weight;
  }

  CUDA_DEVICE static DFFRQuadPhaseExpansion expand(const Constants &source, int component,
                                                   double retarded_time) {
    (void)component;
    DFFRQuadPhaseExpansion expansion;
    expansion.reference_phase =
        gb_phase(source.frequency, source.frequency_derivative,
                 source.frequency_second_derivative, source.initial_phase, retarded_time);
    expansion.instantaneous_frequency =
        gb_frequency(source.frequency, source.frequency_derivative,
                     source.frequency_second_derivative, retarded_time);
    expansion.frequency_rate = gb_frequency_derivative(
        source.frequency_derivative, source.frequency_second_derivative, retarded_time);
    expansion.amplitude_ratio =
        gb_amplitude_ratio(source.frequency, source.frequency_derivative, retarded_time);
    return expansion;
  }

  CUDA_DEVICE static double amplitude_log_slope(const Constants &source, int component,
                                                double midpoint_offset) {
    (void)component;
    return gb_amplitude_log_slope(source.frequency, source.frequency_derivative, midpoint_offset);
  }
};

#endif  // __GB_DFFR_SOURCE_HH__
