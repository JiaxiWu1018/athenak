#ifndef EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
#define EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ps_error.hpp
//  \brief defines an enumerator struct for error types.

namespace Primitive {
enum struct Error {
  SUCCESS,
  RHO_TOO_BIG,
  RHO_TOO_SMALL,
  NANS_IN_CONS,
  NANS_IN_PRIM,
  MAG_TOO_BIG,
  BRACKETING_FAILED,
  NO_SOLUTION,
  CONS_FLOOR,
  PRIM_FLOOR,
  CONS_ADJUSTED,
  LAPSE_TOO_SMALL,
};

struct SolverResult {
  Error error;
  int  iterations;
  bool cons_floor;
  bool prim_floor;
  bool cons_adjusted;
};

struct FloorFlag {
  bool d_floor;
  bool s_floor;
  bool tau_floor;
  bool ye_floor;
};
} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
