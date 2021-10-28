#include <cmath> // floor, ceil

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "openmc/bank.h"
#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/pct.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"


namespace openmc {

//==============================================================================
// PCT setups
//==============================================================================

PCT::PCT() {}

void PCTSimpleSampling::allocate() {
  // Allocate counters for reproducibility
  count.resize(3*simulation::work_per_rank, 0);
}

// Duplicate-Discard (DD)
void PCTDuplicateDiscard::allocate() {
  // Allocate flags. 
  // We need to flag ALL sites since we perform global discard
  discard_flag.resize(3*settings::n_particles, false); // "true" = discarded
  
  // Allocate counters for reproducibility
  count.resize(3*simulation::work_per_rank, 0);
}


//==============================================================================
// Synchronize "global" fission bank
//==============================================================================

void PCT::global_sync( int64_t& start, int64_t& total) const {
  // In order to properly understand the fission bank algorithm, you need to
  // think of the fission and source bank as being one global array divided
  // over multiple processors. At the start, each processor has a random amount
  // of fission bank sites -- each processor needs to know the total number of
  // sites in order to figure out the probability for selecting
  // sites. Furthermore, each proc also needs to know where in the 'global'
  // fission bank its own sites starts in order to ensure reproducibility by
  // skipping ahead to the proper seed.

#ifdef OPENMC_MPI
  start = 0;
  int64_t n_bank = simulation::fission_bank.size();
  MPI_Exscan(&n_bank, &start, 1, MPI_INT64_T, MPI_SUM, mpi::intracomm);

  // While we would expect the value of start on rank 0 to be 0, the MPI
  // standard says that the receive buffer on rank 0 is undefined and not
  // significant
  if (mpi::rank == 0)
    start = 0;

  total = start + n_bank;
  MPI_Bcast(&total, 1, MPI_INT64_T, mpi::n_procs - 1, mpi::intracomm);

#else
  start = 0;
  total = simulation::fission_bank.size();
#endif

  // If there are not that many particles per generation, it's possible that no
  // fission sites were created at all on a single processor. Rather than add
  // extra logic to treat this circumstance, we really want to ensure the user
  // runs enough particles to avoid this in the first place.

  if (simulation::fission_bank.size() == 0) {
    fatal_error(
      "No fission/census sites banked on MPI rank " + std::to_string(mpi::rank));
  }
}

//==============================================================================
// PCT.sample: Sample n_particles from fission_bank and place in sample_bank.
//==============================================================================

// Simple Sampling (SS)
//=====================

void PCTSimpleSampling::sample(int64_t& n_sample) {
  int64_t start, end, total, idx, idx_local;
  n_sample = 0;
  
  // Synchronize "global" fission bank
  global_sync(start, total);
  end = start + simulation::fission_bank.size();
  
  // Make sure all processors start at the same point for random sampling. Then
  // skip ahead in the sequence using the starting index in the 'global'
  // fission bank for each processor.

  int64_t id = simulation::total_gen + overall_generation();
  uint64_t seed = init_seed(id, STREAM_TRACKING);

  // Sample n_particles from fission_bank
  // For reproducibility, we first count how many times each site is sampled
  for (int64_t i = 0; i < settings::n_particles; i++) {
    idx = floor(prn(&seed)*total);

    // Check if it is local
    if (start <= idx && idx < end) {
      idx_local = idx - start;
      count[idx_local]++;
    }
  }

  // Now, we set up the sample bank form the site counts
  for (int64_t i = 0; i < simulation::fission_bank.size(); i++) {
    const auto& site = simulation::fission_bank[i];
    for (int64_t j = 0; j < count[i]; j++) {
      simulation::sample_bank[n_sample] = site;
      ++n_sample;
    }

    // Reset the count
    count[i] = 0;
  }
}

// Duplicate Discard (DD)
//=======================

void PCTDuplicateDiscard::sample(int64_t& n_sample) {
  int64_t start, end, total, idx, idx_local;
  n_sample = 0;
  
  // Synchronize "global" fission bank
  global_sync(start, total);
  end = start + simulation::fission_bank.size();
  
  // Make sure all processors start at the same point for random sampling. Then
  // skip ahead in the sequence using the starting index in the 'global'
  // fission bank for each processor.

  int64_t id = simulation::total_gen + overall_generation();
  uint64_t seed = init_seed(id, STREAM_TRACKING);

  // Duplicate or discard?
  bool duplicate = settings::n_particles > total;

  if (duplicate) {
    // Duplicate

    // Copies needed
    int64_t N_copy = floor(static_cast<double>(settings::n_particles)/total);

    // Additional samples needed (to get exactly n_particles)
    int64_t N_sample = settings::n_particles - N_copy*total;

    // Similar to PCTSimpleSampling, we count the sampled sites first before
    // we set up the sample bank
    
    // Count the copies
    for (int64_t i = 0; i < simulation::fission_bank.size(); i++) {
      count[i] = N_copy;
    }
    
    // Count the additional samples
    for (int64_t i = 0; i < N_sample; i++) {
      idx = floor(prn(&seed)*total);

      // Check if it is local
      if (start <= idx && idx < end) {
        idx_local = idx - start;
        count[idx_local]++;
      }
    }
    
    // Now, we set up the sample bank form the site counts
    for (int64_t i = 0; i < simulation::fission_bank.size(); i++) {
      const auto& site = simulation::fission_bank[i];
      for (int64_t j = 0; j < count[i]; j++) {
        simulation::sample_bank[n_sample] = site;
        ++n_sample;
      }

      // Reset the count
      count[i] = 0;
    }
  } else {
    // Discard

    // Discards needed
    int64_t N_discard = total - settings::n_particles;
    
    // Sample discarded sites and flag it
    for (int64_t i = 0; i < N_discard; i++) {
      while (true) {
        // Sample discard index
        idx = floor(prn(&seed)*total);

        // Flag the site if it is not discarded yet
        if (!discard_flag[idx]) {
          discard_flag[idx] = true;
          break;
        }
        // If the site is already discarded, we resample discard index.
        // In other words, we are performing a rejection sampling.
      }
    }

    // Copy the un-discarded sites
    for (int64_t i = start; i < end; i++) {
      if (!discard_flag[i]) {
        idx_local = i - start;
        const auto& site = simulation::fission_bank[idx_local];
        simulation::sample_bank[n_sample] = site;
        ++n_sample;
      }
    }
    // Reset flag
    for (int64_t i = 0; i < total; i++) {
      discard_flag[i] = false;
    }
  }
}

// Splitting-Roulette (SR) [Sweezy 2014]
//======================================

void PCTSplittingRoulette::sample(int64_t& n_sample) {
  int64_t start, total;
  n_sample = 0;
  
  // Synchronize "global" fission bank
  global_sync(start, total);
  
  // Make sure all processors start at the same point for random sampling. Then
  // skip ahead in the sequence using the starting index in the 'global'
  // fission bank for each processor.

  int64_t id = simulation::total_gen + overall_generation();
  uint64_t seed = init_seed(id, STREAM_TRACKING);
  advance_prn_seed(start, &seed);

  // Perform splitting-roulette to all particles in fission_bank
  for (int64_t i = 0; i < simulation::fission_bank.size(); i++) {
    const auto& site = simulation::fission_bank[i];

    // Survivng probability
    double p_survive = static_cast<double>(settings::n_particles)/total;

    // Splitting
    const int64_t n_survive = std::floor(p_survive);
    for (int64_t j = 0; j < n_survive; ++j) {
      simulation::sample_bank[n_sample] = site;
      ++n_sample;
    }
    p_survive -= static_cast<double>(n_survive);

    // Russian roulette
    if (prn(&seed) < p_survive) {
      simulation::sample_bank[n_sample] = site;
      ++n_sample;
    }
  }
}


// Combing (CO) [Booth 1996]
//==========================

void PCTCombing::sample(int64_t& n_sample) {
  int64_t start, end, total, idx;
  n_sample = 0;
  
  // Synchronize "global" fission bank
  global_sync(start, total);
  end = start + simulation::fission_bank.size();
  
  // Make sure all processors start at the same point for random sampling. 
  int64_t id = simulation::total_gen + overall_generation();
  uint64_t seed = init_seed(id, STREAM_TRACKING);

  // Teeth distance
  double teeth_distance = static_cast<double>(total)/settings::n_particles;

  // Tooth off-set
  double offset = prn(&seed)*teeth_distance;

  // First and last hiting tooth
  int64_t tooth_start = std::ceil((start-offset)/teeth_distance);
  int64_t tooth_end   = std::floor((end-offset)/teeth_distance) + 1;

  // Comb particles in fission_bank
  double tooth = tooth_start*teeth_distance + offset;
  for (int64_t i = tooth_start; i < tooth_end; i++) {
    idx = floor(tooth) - start;
    const auto& site = simulation::fission_bank[idx];
    simulation::sample_bank[n_sample] = site;
    ++n_sample;
    
    // Next tooth
    tooth += teeth_distance;
  }
}


// New Combing (COX) [Booth 1996]
//===============================

void PCTNewCombing::sample(int64_t& n_sample) {
  int64_t start, end, total, idx;
  n_sample = 0;
  
  // Synchronize "global" fission bank
  global_sync(start, total);
  end = start + simulation::fission_bank.size();
 
  // Teeth distance
  double teeth_distance = static_cast<double>(total)/settings::n_particles;

  // First and last possible hitting tooth index
  int64_t tooth_start = floor(start/teeth_distance);
  int64_t tooth_end   = ceil(end/teeth_distance);

  // Make sure all processors start at the same point for random sampling. Then
  // skip ahead in the sequence using the starting tooth index in the 'global'
  // combing for each processor.
  int64_t id = simulation::total_gen + overall_generation();
  uint64_t seed = init_seed(id, STREAM_TRACKING);
  advance_prn_seed(tooth_start, &seed);

  // Comb particles in fission_bank
  double tooth;
  for (int64_t i = tooth_start; i < tooth_end; i++) {
    tooth = (i + prn(&seed))*teeth_distance;

    // Check if it is local
    if (tooth >= start && tooth < end) {
      idx = floor(tooth) - start;
      const auto& site = simulation::fission_bank[idx];
      simulation::sample_bank[n_sample] = site;
      ++n_sample;
    }
  }
}
} // namespace openmc
