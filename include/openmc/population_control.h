#ifndef OPENMC_PCT_H
#define OPENMC_PCT_H

#include "openmc/settings.h"
#include "openmc/vector.h"


namespace openmc {

//==============================================================================
// Abstract class for Population Control Technique (PCT)
//==============================================================================

class PCT {
protected:
  std::string label_;
  void global_sync(int64_t& start, int64_t& total) const;

public:
  PCT();
  virtual ~PCT() {};
  virtual void sample(int64_t& n_sample) = 0;
  virtual void allocate() = 0;

  // Getters
  std::string label() const {return label_;}
};


//==============================================================================
// Simple Sampling
//==============================================================================

class PCTSimpleSampling : public PCT {
private:
    vector<int64_t> count; // for the source sites

public:
  PCTSimpleSampling() {label_="SS";}
  void sample(int64_t& n_sample);
  void allocate();
};


//==============================================================================
// Duplicate-Discard [Leppanen 2013]
//==============================================================================

class PCTDuplicateDiscard : public PCT {
private:
  vector<bool> discard_flag; // for the source sites
  vector<int64_t> count; 

public:
  PCTDuplicateDiscard() {label_="DD";}
  void sample(int64_t& n_sample);
  void allocate();
};


//==============================================================================
// Splitting-Roulette [Sweezy 2014]
//==============================================================================

class PCTSplittingRoulette : public PCT {
public:
  PCTSplittingRoulette() {label_="SR";}
  
  void sample(int64_t& n_sample);
  void allocate() {;}
};


//==============================================================================
// Combing [Booth 1996]
//==============================================================================

class PCTCombing : public PCT {
public:
  PCTCombing() {label_="CO";}
  void sample(int64_t& n_sample);
  void allocate() {;}
};


//==============================================================================
// New Combing [Ajami 2021]
//==============================================================================

class PCTNewCombing : public PCT {
public:
  PCTNewCombing() {label_="COX";}
  void sample(int64_t& n_sample);
  void allocate() {;}
};

} // namespace openmc

#endif // OPENMC_PCT_H
