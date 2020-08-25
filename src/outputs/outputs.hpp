#ifndef OUTPUTS_OUTPUTS_HPP_
#define OUTPUTS_OUTPUTS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outputs.hpp
//  \brief provides classes to handle ALL types of data output

//#include <cstdio>  // std::size_t
#include <string>
#include <vector>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "io_wrapper.hpp"

// forward declarations
class Mesh;
class ParameterInput;
class Coordinates;

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief  container for parameters read from <output> block in the input file

struct OutputParameters {
  int block_number;
  std::string block_name;
  std::string file_basename;
  std::string file_id;
  std::string variable;
  std::string file_type;
  std::string data_format;
  Real last_time, dt;
  int file_number;
  bool include_gzs;
  bool slice1, slice2, slice3;
  Real slice_x1, slice_x2, slice_x3;
};

//----------------------------------------------------------------------------------------
//! \struct OutputData
//  \brief container for output data and metadata; node in std::list

struct OutputData {
  std::string type;        // one of (SCALARS,VECTORS) used for vtk outputs
  std::string name;
  AthenaArray<Real> cc_data;  // array containing data (deep copied)
};

//----------------------------------------------------------------------------------------
// \brief abstract base class for different output types (modes/formats); node in
//        std::list of OutputType created & stored in the Outputs class

class OutputType {
 public:
  OutputType(OutputParameters oparams, Mesh *pm);
  virtual ~OutputType() = default;
  // copy constructor and assignment operator
  OutputType(const OutputType& copy_other) = default;
  OutputType& operator=(const OutputType& copy_other) = default;
  // move constructor and assignment operator
  OutputType(OutputType&&) = default;
  OutputType& operator=(OutputType&&) = default;

  // data
  int nout1, nout2, nout3;           // dimensions of output arrays for this type
  int ois, ojs, oks;                 // starting indices of data to be output
  Real x1posn, x2posn, x3posn;
  OutputParameters output_params;    // data read from <output> block for this type

  // functions
  void LoadOutputData(Mesh *pm);
  // following pure virtual function must be implemented in all derived classes
  virtual void WriteOutputFile(Mesh *pm) = 0;

 protected:
  std::list<OutputData> data_list_;
};

//----------------------------------------------------------------------------------------
//! \class FormattedTableOutput
//  \brief derived OutputType class for formatted table (tabular) data

class FormattedTableOutput : public OutputType {
 public:
  FormattedTableOutput(OutputParameters oparams, Mesh *pm);
  void WriteOutputFile(Mesh *pm) override;
};

//----------------------------------------------------------------------------------------
//! \class Outputs

//  \brief root class for all Athena++ outputs. Provides a singly linked list of
//  OutputTypes, with each node representing one mode/format of output to be made.

class Outputs {
 public:
  Outputs(std::unique_ptr<ParameterInput> &pin, Mesh *pm);
  ~Outputs();

  // use vector of pointers to OutputTypes since it is an abstract base class 
  std::vector<OutputType*> poutput_list_;  

 private:

};

#endif // OUTPUTS_OUTPUTS_HPP_
